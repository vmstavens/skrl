import time
from dataclasses import dataclass

import mujoco as mj
import mujoco.viewer
import numpy as np
import tyro

from testing.envs import mocap_cable


@dataclass
class ViewerArgs:
    seed: int = 0
    realtime: bool = True
    reset_every: int = 0


def _randomize_state(
    model: mj.MjModel, data: mj.MjData, rng: np.random.Generator
) -> None:
    cfg = mocap_cable.default_config()
    goal_low = np.array([-0.3, -0.3, 0.2], dtype=np.float64)
    goal_high = np.array([0.3, 0.3, 0.4], dtype=np.float64)

    goal = rng.uniform(goal_low, goal_high)

    mocap_body_id = model.body("mocap").id
    ee_mocap_id = model.body("mocap").mocapid
    goal_mocap_id = model.body("goal").mocapid

    mocap_home_pos = np.array(model.body_pos[mocap_body_id], dtype=np.float64)
    mocap_home_quat = np.array(model.body_quat[mocap_body_id], dtype=np.float64)

    data.mocap_pos[ee_mocap_id] = mocap_home_pos
    data.mocap_quat[ee_mocap_id] = mocap_home_quat
    data.mocap_pos[goal_mocap_id] = goal
    data.mocap_quat[goal_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])

    if data.ctrl.size:
        data.ctrl[0] = float(cfg.grip_ctrl_min)


def main() -> None:
    args = tyro.cli(ViewerArgs)

    model = mocap_cable._build_model()
    data = mj.MjData(model)

    rng = np.random.default_rng(args.seed)
    _randomize_state(model, data, rng)
    mj.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
        viewer.cam.azimuth = model.vis.global_.azimuth
        viewer.cam.elevation = model.vis.global_.elevation
        viewer.cam.lookat = model.stat.center
        viewer.cam.distance = model.stat.extent

        step_count = 0
        while viewer.is_running():
            step_start = time.time()

            mj.mj_step(model, data)
            viewer.sync()

            step_count += 1
            if args.reset_every and step_count % args.reset_every == 0:
                mj.mj_resetData(model, data)
                _randomize_state(model, data, rng)
                mj.mj_forward(model, data)

            if args.realtime:
                time_until_next_step = model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
