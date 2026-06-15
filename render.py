import time

import jax
import jax.numpy as jnp
import mediapy as media

from testing.envs.pipe_insert_2 import PipeInsert2

print("1")
env = PipeInsert2()
print("2")

# jit_reset = env.reset
# jit_step = env.step
jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

print("3")
state = jit_reset(jax.random.PRNGKey(0))
print("3.5")
state = jax.block_until_ready(state)

rollout = [state]
print("4")

f = 0.5
for i in range(10):
    action = jnp.array(
        [
            jnp.sin(state.data.time * 2 * jnp.pi * f + j * 2 * jnp.pi / env.action_size)
            for j in range(env.action_size)
        ],
        dtype=jnp.float32,
    )

    state = jit_step(state, action)

    rollout.append(state)

# Move rollout to host memory before rendering
rollout_host = jax.device_get(rollout)

# Render frames
frames = env.render(rollout_host)

# Write video to file
output_path = "pipe_insert_rollout.mp4"
media.write_video(output_path, frames, fps=1.0 / env.dt)

print(f"Video saved to {output_path}")
