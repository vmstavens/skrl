from testing.experiments.pipe_insert.exp_utils import *  # noqa: F401,F403

from testing.envs.pipe_insert_2_warp import PipeInsert2Warp
from testing.wrappers_warp import WarpWrapper


def setup_environment(
    batch_size: int = 100,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
    version: int = 2,
):
    if version != 2:
        raise ValueError("Warp wrapper currently supports only PipeInsert2 (version=2).")

    env = PipeInsert2Warp(
        batch_size=batch_size,
        episode_length=episode_length,
        action_repeat=action_repeat,
    )

    # auto_reset handled by the trainer; keep parity with MJX stack
    return WarpWrapper(env)
