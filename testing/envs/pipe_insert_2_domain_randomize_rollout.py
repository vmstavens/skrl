from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from testing.envs.pipe_insert_2 import PipeInsert2, domain_randomize

DEFAULT_OUTPUT_DIR = Path(
    "testing/experiments/pipe_insert/media/domain_randomized_rollouts"
)


def main() -> None:

    env = PipeInsert2()
    mj_model = env.mj_model
    base_model = mjx.put_model(mj_model, impl="jax")
    env._mjx_model = base_model
    action_size = env.action_size

    sequence_actions = np.asarray(env.sequence_actions)
    sequence_len = int(sequence_actions.shape[0]) if sequence_actions.ndim > 0 else 0

    args.output_dir.mkdir(parents=True, exist_ok=True)

    master_key = jax.random.PRNGKey(args.seed)
    model_key = jax.random.fold_in(master_key, args.index * 3)
    reset_key = jax.random.fold_in(master_key, args.index * 3 + 1)
    action_key = jax.random.fold_in(master_key, args.index * 3 + 2)

    model_batch, in_axes = domain_randomize(
        base_model, mj_model, jax.random.split(model_key, 1)
    )
    env._mjx_model = _unbatch_model(model_batch, in_axes)

    state = _reset_state(env, reset_key, impl=args.impl)

    video_path = args.output_dir / f"{args.prefix}_{args.index:03d}.mp4"
    with imageio.get_writer(video_path, fps=args.fps) as writer:
        frame = _render_frame(renderer, mj_model, state, cam_id)
        writer.append_data(frame)

        for step in range(args.num_steps):
            if args.action_mode == "sequence" and step < sequence_len:
                action = jp.asarray(sequence_actions[step], dtype=jp.float32)
            elif args.action_mode == "random":
                action_key, key = jax.random.split(action_key)
                action = jax.random.uniform(
                    key,
                    shape=(action_size,),
                    minval=-args.random_action_scale,
                    maxval=args.random_action_scale,
                )
            else:
                action = jp.zeros((action_size,), dtype=jp.float32)

            state = _step_state(env, state, action)
            frame = _render_frame(renderer, mj_model, state, cam_id)
            writer.append_data(frame)

            if bool(np.array(state.done)):
                break

    print(f"Saved rollout video to {video_path}")


if __name__ == "__main__":
    main()
