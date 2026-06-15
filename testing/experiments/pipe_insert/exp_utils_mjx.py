from testing.experiments.pipe_insert.exp_utils import *  # noqa: F401,F403

from testing.envs.pipe_insert_2_mjx import PipeInsert2


def setup_environment(
    batch_size: int = 100,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
    version: int = 2,
):
    """Set up the MJX environment with the original MJX-based PipeInsert2."""

    versions = [1, 2]
    if version not in versions:
        raise ValueError(f"Invalid version {version} expected {versions}")

    _env_versions = [PipeInsert, PipeInsert2]
    env = _env_versions[version - 1]()

    env = wrap.create(
        env,
        batch_size=batch_size,
        episode_length=episode_length,
        auto_reset=auto_reset,
        action_repeat=action_repeat,
    )
    env = wrap_env(env, wrapper="playground")

    return env
