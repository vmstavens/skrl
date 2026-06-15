from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

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

    def cb(self, key: int) -> None:
        if key == glfw.KEY_SPACE:
            np.save(self.keys["qpos"], self.d.qpos)
            np.save(self.keys["qvel"], self.d.qvel)
            np.save(self.keys["ctrl"], self.d.ctrl)
        if key == glfw.KEY_PERIOD:
            self.d.qpos = np.load(self.keys["qpos"])
            self.d.qvel = np.load(self.keys["qvel"])
            self.d.ctrl = np.load(self.keys["ctrl"])
        if key == glfw.KEY_ENTER:
            print("perform diffusion policy")
            pass

    def get_obs(self) -> torch.Tensor:
        T_w_keypoint_1 = get_pose(self.m, self.d, "keypoint_1", ObjType.SITE)
        T_w_keypoint_2 = get_pose(self.m, self.d, "keypoint_2", ObjType.SITE)
        T_w_keypoint_3 = get_pose(self.m, self.d, "keypoint_3", ObjType.SITE)
        T_w_target_1 = get_pose(self.m, self.d, "target_1", ObjType.SITE)
        T_w_target_2 = get_pose(self.m, self.d, "target_2", ObjType.SITE)
        T_w_target_3 = get_pose(self.m, self.d, "target_3", ObjType.SITE)

        d1 = np.linalg.norm(T_w_target_1.t - T_w_keypoint_1.t)
        d2 = np.linalg.norm(T_w_target_2.t - T_w_keypoint_2.t)
        d3 = np.linalg.norm(T_w_target_3.t - T_w_keypoint_3.t)

        d = np.array([d1, d2, d3])
        v = np.array([0, 0, 0])

        o = np.concatenate([d, v])

        return torch.Tensor(o)

    def run(self, m, d):
        self.m = m
        self.d = d
        self.d.qpos = np.load(self.keys["qpos"])
        self.d.qvel = np.load(self.keys["qvel"])
        self.d.ctrl = np.load(self.keys["ctrl"])

        o = self.get_obs()
        print(o)

        queue = deque(
            [o] * self.agent.config["obs_horizon"],
            maxlen=self.agent.config["obs_horizon"],
        )

        print(queue)

        # _o = torch.tensor(queue)
        obs_tensor = torch.unsqueeze(torch.stack(list(queue), dim=0), dim=0).to(
            self.env.device
        )

        print(obs_tensor.shape)

        a, _, _ = self.agent.act(obs_tensor)

        with mujoco.viewer.launch_passive(m, d, key_callback=self.cb) as viewer:
            while True:
                step_start = time.time()
                mj.mj_step(m, d)
                viewer.sync()

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
    print("done...")
    sim = Test(agent, env)
    sim.run(m, d)


if __name__ == "__main__":
    main()
