"""Replay a recorded demonstration in a loop using the raw MuJoCo simulation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco as mj
import mujoco.viewer
import numpy as np

from testing.envs.pipe_insert import PipeInsert


def load_demo(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Demo file not found: {path}")
    with path.open() as f:
        return json.load(f)


def find_latest_demo(data_dir: Path) -> Path:
    demos = sorted(data_dir.glob("*.json"))
    if not demos:
        raise FileNotFoundError(f"No demo files found in {data_dir}")
    return demos[-1]


class DemoReplayer:
    def __init__(self, model: mj.MjModel, data: mj.MjData, actions: np.ndarray):
        self.m = model
        self.d = data
        self.actions = actions  # sequence of delta ctrl, shape (T, 2)

        # Initial state for reset
        self.qpos0 = np.load("testing/experiments/pipe_insert/constants/qpos_free.npy")
        self.qvel0 = np.load("testing/experiments/pipe_insert/constants/qvel_free.npy")
        self.ctrl0 = np.load("testing/experiments/pipe_insert/constants/ctrl_free.npy")

        self.ctrl_low = self.m.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.m.actuator_ctrlrange[:, 1]
        self.idx = 0

        # Seed state
        self.reset_state()

    def reset_state(self) -> None:
        self.d.qpos[:] = self.qpos0
        self.d.qvel[:] = self.qvel0
        self.d.ctrl[:] = self.ctrl0
        mj.mj_forward(self.m, self.d)
        self.idx = 0

    def step_action(self) -> None:
        delta = self.actions[self.idx]
        print(delta)
        ctrl = self.d.ctrl.copy()
        ctrl[:2] = np.clip(ctrl[:2] + delta, self.ctrl_low, self.ctrl_high)
        self.d.ctrl[:] = ctrl
        self.idx += 1
        if self.idx >= len(self.actions):
            self.idx = 0
            self.reset_state()

    def run(self) -> None:
        with mujoco.viewer.launch_passive(self.m, self.d) as viewer:
            while viewer.is_running():
                step_start = time.time()
                self.step_action()
                mj.mj_step(self.m, self.d)
                viewer.sync()

                # real-time pacing
                time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a recorded demo in a loop.")
    parser.add_argument(
        "--demo",
        type=Path,
        default=None,
        help="Path to demo JSON (default: latest in data_collection/).",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).resolve().parent.parent / "data_collection"
    demo_path = args.demo if args.demo is not None else find_latest_demo(data_dir)
    demo = load_demo(demo_path)
    actions = np.asarray(demo["actions"], dtype=np.float32)

    env = PipeInsert()
    model = env.mj_model
    data = mj.MjData(model)

    print(f"Replaying demo {demo_path.name} ({len(actions)} steps) in a loop.")
    DemoReplayer(model, data, actions).run()


if __name__ == "__main__":
    main()
