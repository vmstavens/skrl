from __future__ import annotations

import time
from collections import deque

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
import torch

from testing.envs.pipe_insert import PipeInsert
from testing.experiments.pipe_insert.exp_utils import get_dp_config, setup_environment
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


def build_models(env):
    """Create DP models sized to the pipe insert environment."""
    dp_config = get_dp_config(exp_name="sim_dp", env=env, wandb=False)

    a_dim = env.action_space.shape[0]
    o_dim = env.observation_space.shape[0]
    device = env.device

    model = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(device)
    ema_model = ConditionalUnet1D(a_dim=a_dim, o_dim=o_dim, config=dp_config).to(device)
    ema = EMAModel(model.parameters(), power=dp_config["ema_power"])

    return {"model": model, "ema_model": ema_model}, ema, dp_config


class Test:
    def __init__(self, agent: DiffusionPolicy, env):
        self.keys = {
            "qpos": "testing/experiments/pipe_insert/constants/qpos_free.npy",
            "qvel": "testing/experiments/pipe_insert/constants/qvel_free.npy",
            "ctrl": "testing/experiments/pipe_insert/constants/ctrl_free.npy",
        }
        self.agent = agent
        self.env = env

        # Rolling buffer of observations for DP
        self.obs_horizon = int(self.agent.config["obs_horizon"])
        self.queue: deque[torch.Tensor] | None = None

        # Policy execution state
        self.pending_policy = False
        self.action_plan: torch.Tensor | None = (
            None  # [T, a_dim] or [1,T,a_dim] depending on agent
        )
        self.action_idx = 0
        self.last_act_time = 0.0

        # Optional: act at a lower rate than physics
        self.control_dt = float(self.agent.config.get("control_dt", 0.02))  # seconds
        self.device = self.env.device

    def cb(self, key: int) -> None:
        # NOTE: viewer key_callback calls with raw key codes; we only use key here.
        if key == glfw.KEY_SPACE:
            np.save(self.keys["qpos"], self.d.qpos.copy())
            np.save(self.keys["qvel"], self.d.qvel.copy())
            np.save(self.keys["ctrl"], self.d.ctrl.copy())
            print("Saved qpos/qvel/ctrl.")

        elif key == glfw.KEY_PERIOD:
            self.d.qpos[:] = np.load(self.keys["qpos"])
            self.d.qvel[:] = np.load(self.keys["qvel"])
            self.d.ctrl[:] = np.load(self.keys["ctrl"])
            mj.mj_forward(self.m, self.d)
            print("Loaded qpos/qvel/ctrl.")

        elif key == glfw.KEY_ENTER:
            print("Queued diffusion policy rollout (will start on next sim loop).")
            self.pending_policy = True

        elif key == glfw.KEY_ESCAPE:
            # viewer handles close; we can just print
            print("ESC pressed.")

    def get_obs(self) -> torch.Tensor:
        # Your handcrafted obs: distances between keypoints and targets.
        T_w_keypoint_1 = get_pose(self.m, self.d, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.m, self.d, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.m, self.d, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.m, self.d, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.m, self.d, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.m, self.d, "target_3", ObjType.SITE)

        d1 = np.linalg.norm(T_w_target_1.t - T_w_keypoint_1.t)
        d2 = np.linalg.norm(T_w_target_2.t - T_w_keypoint_2.t)
        d3 = np.linalg.norm(T_w_target_3.t - T_w_keypoint_3.t)

        d = np.array([d1, d2, d3], dtype=np.float32)
        v = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        o = np.concatenate([d, v]).astype(np.float32)
        return torch.from_numpy(o)  # [o_dim]

    def _init_queue_if_needed(self) -> None:
        if self.queue is None:
            o = self.get_obs()
            self.queue = deque([o] * self.obs_horizon, maxlen=self.obs_horizon)

    def _update_obs_queue(self) -> None:
        self._init_queue_if_needed()
        assert self.queue is not None
        self.queue.append(self.get_obs())

    def _queue_to_tensor(self) -> torch.Tensor:
        """
        Returns obs tensor of shape [num_envs, obs_horizon, o_dim].
        Here num_envs=1, so: [1, H, o_dim]
        """
        assert self.queue is not None
        obs = torch.stack(list(self.queue), dim=0)  # [H, o_dim]
        obs = obs.unsqueeze(0).to(self.device)  # [1, H, o_dim]
        return obs

    def _compute_action_plan(self) -> None:
        """
        Runs diffusion policy once, stores the returned action sequence.
        Supports either:
          - agent.act(obs) -> (a, _, _), where a is [1, T, a_dim] or [T, a_dim] or [1, a_dim]
        """
        obs_tensor = self._queue_to_tensor()

        with torch.no_grad():
            a, _, _ = self.agent.act(obs_tensor)

        # Normalize action plan to [T, a_dim]
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)

        if torch.is_tensor(a):
            a = a.detach()

        if a.ndim == 3:
            # [B, T, A]
            a_plan = a[0]
        elif a.ndim == 2:
            # [T, A]
            a_plan = a
        elif a.ndim == 1:
            # [A] single step
            a_plan = a.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected action tensor shape: {tuple(a.shape)}")

        self.action_plan = a_plan.to(self.device)
        self.action_idx = 0
        self.last_act_time = self.d.time
        print(f"Computed action plan: {tuple(self.action_plan.shape)} (T, a_dim)")

    def _apply_action(self, action: torch.Tensor) -> None:
        """
        Apply action to MuJoCo controls.
        Assumes action is [a_dim] in the correct control space for your model.
        """
        a_np = action.detach().to("cpu").numpy().astype(np.float64, copy=False)
        if a_np.shape[0] != self.d.ctrl.shape[0]:
            raise ValueError(
                f"Action dim mismatch: got {a_np.shape[0]}, expected {self.d.ctrl.shape[0]}"
            )
        self.d.ctrl[:] += a_np

    def run(self, m, d):
        self.m = m
        self.d = d

        # Load initial state
        self.d.qpos[:] = np.load(self.keys["qpos"])
        self.d.qvel[:] = np.load(self.keys["qvel"])
        self.d.ctrl[:] = np.load(self.keys["ctrl"])
        mj.mj_forward(self.m, self.d)

        # Prime obs queue
        self._init_queue_if_needed()
        assert self.queue is not None
        print("Initial obs:", self.queue[-1])
        print("Initial obs tensor shape:", tuple(self._queue_to_tensor().shape))

        with mujoco.viewer.launch_passive(m, d, key_callback=self.cb) as viewer:
            while viewer.is_running():
                step_start = time.time()

                # Always update obs history each sim step (or you can throttle if desired)
                self._update_obs_queue()

                # On ENTER: compute a new action plan at the next loop iteration
                if self.pending_policy:
                    self.pending_policy = False
                    self._compute_action_plan()

                # If we have an active plan, apply actions sequentially
                if self.action_plan is not None:
                    # Optionally hold each planned action for control_dt seconds
                    if (self.d.time - self.last_act_time) >= self.control_dt:
                        if self.action_idx < self.action_plan.shape[0]:
                            self._apply_action(self.action_plan[self.action_idx])
                            self.action_idx += 1
                            self.last_act_time = self.d.time
                        else:
                            # Finished plan
                            self.action_plan = None
                            print("Finished action plan.")

                mj.mj_step(m, d)
                viewer.sync()

                # real-time pacing
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main():
    torch.set_grad_enabled(False)

    _env = PipeInsert()
    env = setup_environment(batch_size=1)

    models, ema, dp_config = build_models(env)

    o_dim = _env.observation_size
    a_dim = _env.action_size

    print(
        f"Initialized DP models with obs_dim={dp_config['obs_dim']} "
        f"action_dim={env.action_space.shape[0]}"
    )
    print(f"EMA shadow params: {len(ema.shadow_params)} tensors")

    agent = DiffusionPolicy(
        a_dim=a_dim, o_dim=o_dim, models=models, ema=ema, config=dp_config
    )

    agent.load(
        "testing/experiments/pipe_insert/.runs/data_100/models/model_epoch_40.pth"
    )

    m = _env.mj_model
    d = mj.MjData(m)

    sim = Test(agent, env)
    sim.run(m, d)


if __name__ == "__main__":
    main()
