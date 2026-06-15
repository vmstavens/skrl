from __future__ import annotations

from collections import deque
from pathlib import Path

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
import torch
from mujoco import mjx

from testing.envs.pipe_insert_2 import PipeInsert2
from testing.shen.diffusion_policy_state import DiffusionPolicy

CHECKPOINT_PATH = Path(
    "testing/experiments/pipe_insert/.runs/data_100_2_1_action_horizon/models/latest_model.pth"
)
OUTPUT_VIDEO = Path("testing/experiments/pipe_insert/media/dp_mjx_rollout.mp4")
NUM_STEPS = 1000
FPS = 30
SEED = 0
NUM_DIFFUSION_ITERS = 10


def _render_frame(
    renderer: mj.Renderer, model: mj.MjModel, state, cam_id: int
) -> np.ndarray:
    data = mjx.get_data(model, state.data)[0]
    renderer.update_scene(data=data, camera=cam_id)
    return renderer.render()


def main() -> None:
    env = PipeInsert2()
    a_dim = env.action_size
    o_dim = env.observation_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dp_policy = DiffusionPolicy.load(
        CHECKPOINT_PATH.as_posix(),
        a_dim=a_dim,
        o_dim=o_dim,
        device=device,
    )
    dp_policy._num_diffusion_iters = NUM_DIFFUSION_ITERS
    dp_policy._act_horizon = 1

    obs_horizon = dp_policy.config.get("obs_horizon", 1)

    rng = jax.random.PRNGKey(SEED)
    state = env.reset(rng)

    obs = np.array(state.obs)
    obs_deque = deque([torch.from_numpy(obs)] * obs_horizon, maxlen=obs_horizon)

    renderer = mj.Renderer(env.mj_model, width=640, height=480)
    cam_id = env.mj_model.cam("cam").id if env.mj_model.ncam > 0 else -1

    frames: list[np.ndarray] = [_render_frame(renderer, env.mj_model, state, cam_id)]

    for _ in range(NUM_STEPS):
        obs_seq = (
            torch.stack(list(obs_deque)).unsqueeze(0).to(device, dtype=torch.float32)
        )
        with torch.no_grad():
            actions, _, _ = dp_policy.act(obs_seq)
        start = max(obs_horizon - 1, 0)
        action = actions[0, start, :].detach().cpu().numpy()
        action_jax = jp.array(action, dtype=jp.float32)

        state = env.step(state, action_jax)
        obs_deque.append(torch.from_numpy(np.array(state.obs)))
        frames.append(_render_frame(renderer, env.mj_model, state, cam_id))

        if bool(np.array(state.done)):
            break

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(OUTPUT_VIDEO, frames, fps=FPS)
    print(f"Saved MJX DP rollout video to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()
