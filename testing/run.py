import jax
import jax.numpy as jp
from mujoco_playground import registry

from testing.envs.xpose import XPose, default_config


def setup_environment(env_name: str = "xpose"):
    """Set up the XPose environment without wrappers."""
    registry.dm_control_suite.register_environment(
        env_name=env_name,
        env_class=XPose,
        cfg_class=default_config,
    )
    env = registry.dm_control_suite.load(env_name)
    return env


def main():
    """Simple rollout with random actions - no rendering."""
    # Setup environment
    print("Setting up environment...")
    env = setup_environment("xpose")

    # Initialize the environment
    rng = jax.random.PRNGKey(42)
    state = env.reset(rng)

    num_timesteps = 1000

    print("Starting simulation loop...")

    for i in range(num_timesteps):
        # Generate random action
        rng, action_rng = jax.random.split(rng)
        action = jax.random.uniform(
            action_rng, shape=(env.action_size,), minval=-1.0, maxval=1.0
        )

        # Step the environment
        state = env.step(state, action)

        # Print simple debug info
        print(f"Step {i}: Reward = {state.reward:.3f}, Done = {state.done}")

        # Reset if episode ended
        if state.done:
            print(f"Episode ended at step {i}, resetting...")
            rng, reset_rng = jax.random.split(rng)
            state = env.reset(reset_rng)

    print("Simulation completed!")


if __name__ == "__main__":
    main()
