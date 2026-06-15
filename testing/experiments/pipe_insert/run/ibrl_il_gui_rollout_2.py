from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np
import torch

from testing.envs.pipe_insert_2 import PipeInsert2
from testing.experiments.pipe_insert.exp_utils import (
    exp_set_seed,
    get_drlr_config,
    get_expert_memory_2,
    get_memory,
    get_td3_models,
    setup_environment,
)
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import DiffusionPolicy
from testing.shen.ibrl_sac_o_o2_clean_test import IBRL

CHECKPOINT_PATH = Path(
    "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon/models/latest_model.pth"
)
EXPERT_DATA_DIR = "testing/experiments/pipe_insert/run/data_collection_2"
NUM_STEPS = 2500
NUM_DIFFUSION_ITERS = 5
ACTION_INTERVAL_STEPS = 1  # set to 150 to match dp_data_collection_rollout_2


class IBRLGuiRollout:
    def __init__(self) -> None:
        exp_set_seed()

        self._mjx_env = PipeInsert2()
        self.m = self._mjx_env.mj_model
        self.d = mj.MjData(self.m)

        self.keys = self._mjx_env.keys

        self.ctrl_low = self.m.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.m.actuator_ctrlrange[:, 1]

        self.qpos0_gripper = None
        self.qpos0_cable = None
        self.ctrl0 = None

        self.agent = self._build_agent()
        self.device = self.agent.device
        self.obs_history = deque(
            maxlen=self.agent.IL_policy.config.get("obs_horizon", 1)
        )
        self.action_cache = None
        self.step_idx = 0

    def _build_agent(self) -> IBRL:
        date_time = datetime.now()
        time_stamp = date_time.strftime("%Y%m%d_%H_%M_%S")
        exp_name = Path(__file__).stem + f"_il_gui_{time_stamp}"

        env = setup_environment(batch_size=1, version=2)
        drlr_config = get_drlr_config(exp_name, env, wandb=False)
        drlr_config["actor"] = "il"

        # Rollout-only: disable training updates to avoid sampling empty memory.
        drlr_config["gradient_steps"] = 0
        drlr_config["learning_starts"] = NUM_STEPS + 1

        dp_policy = DiffusionPolicy.load(
            CHECKPOINT_PATH.as_posix(),
            a_dim=env.action_space.shape[0],
            o_dim=env.observation_space.shape[0],
            device=env.device,
        )
        dp_policy._num_diffusion_iters = NUM_DIFFUSION_ITERS
        dp_policy._act_horizon = 1

        il_models = {"policy": dp_policy}
        rl_models = get_td3_models(env)

        expert_memory = get_expert_memory_2(
            expert_data_dir=EXPERT_DATA_DIR,
            states_label="states_dense",
            actions_label="actions",
            rewards_label="rewards_dense",
            next_states_label="next_states_dense",
            dones_label="dones",
        )
        memory = get_memory(env, capacity=100_000)

        agent = IBRL(
            models=rl_models,
            models_il=il_models,
            memory=memory,
            expert_memory=expert_memory,
            cfg=drlr_config,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
        )
        agent.init(trainer_cfg={"timesteps": NUM_STEPS})
        agent.set_mode("eval")
        return agent

    def get_obs(self) -> np.ndarray:
        T_w_keypoint_1 = get_pose(self.m, self.d, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.m, self.d, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.m, self.d, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.m, self.d, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.m, self.d, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.m, self.d, "target_3", ObjType.SITE)

        d1 = T_w_target_1.t - T_w_keypoint_1.t
        d2 = T_w_target_2.t - T_w_keypoint_2.t
        d3 = T_w_target_3.t - T_w_keypoint_3.t
        return np.concatenate([d1, d2, d3], axis=0).astype(np.float32)

    def randomize_state(self) -> None:
        if self.qpos0_gripper is None or self.qpos0_cable is None or self.ctrl0 is None:
            raise RuntimeError("Base state not initialized. Call run() first.")
        delta = np.random.uniform(-0.3, 0.3, 3)
        delta = np.zeros(3)
        self.d.qpos[:3] = self.qpos0_gripper + delta
        self.d.ctrl[:3] = self.ctrl0 + delta
        self.d.qpos[3:6] = self.qpos0_cable + delta
        self.d.qvel[:] = 0.0

    def _select_action(self, obs: np.ndarray) -> np.ndarray:
        states = torch.from_numpy(obs).to(self.device).unsqueeze(0)
        self.agent.pre_interaction(states, self.step_idx, NUM_STEPS)
        with torch.no_grad():
            actions, _, _ = self.agent.act(states, self.step_idx, NUM_STEPS)
        action = actions[0].detach().cpu().numpy()
        return action
        # return action * ACTION_SCALE

    def run(self) -> None:
        self.d.qpos[:] = np.load(self.keys["qpos"])
        self.d.qvel[:] = np.load(self.keys["qvel"])
        self.d.ctrl[:] = np.load(self.keys["ctrl"])

        self.qpos0_gripper = self.d.qpos[:3].copy()
        self.qpos0_cable = self.d.qpos[3:6].copy()
        self.ctrl0 = self.d.ctrl[:3].copy()

        self.randomize_state()
        mj.mj_forward(self.m, self.d)

        data = {"states": [], "actions": []}

        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running() and self.step_idx < NUM_STEPS + 100:
                step_start = time.time()

                if self.step_idx > 2500:
                    with open(
                        "testing/experiments/pipe_insert/tmp/success_rollout_ibrl_2.json",
                        "w",
                    ) as f:
                        json.dump(data, f, indent=4)
                    print("saving...")
                    input()

                obs = self.get_obs()
                if len(self.obs_history) == 0:
                    for _ in range(self.agent.IL_policy.config.get("obs_horizon", 1)):
                        self.obs_history.append(torch.from_numpy(obs))
                obs_seq = torch.stack(list(self.obs_history)).unsqueeze(0)
                # if self.action_cache is None or (
                #     self.step_idx % ACTION_INTERVAL_STEPS == 0
                # ):
                self.action_cache = self._select_action(obs)
                done = np.linalg.norm(self.get_obs()) < 0.04

                print(done, self.step_idx)

                self.d.ctrl[:3] = np.clip(
                    self.d.ctrl[:3] + self.action_cache,
                    self.ctrl_low,
                    self.ctrl_high,
                )
                mj.mj_step(self.m, self.d)
                viewer.sync()

                data["actions"].append(self.action_cache.tolist())
                data["states"].append(obs_seq.tolist())

                next_obs = self.get_obs()
                next_states = torch.from_numpy(next_obs).to(self.device).unsqueeze(0)
                self.obs_history.append(torch.from_numpy(next_obs))
                self.agent.post_interaction(next_states, self.step_idx, NUM_STEPS)

                self.step_idx += 1

                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main() -> None:
    sim = IBRLGuiRollout()
    sim.run()


if __name__ == "__main__":
    main()
