from __future__ import annotations

from enum import Enum
from typing import Union

import mujoco as mj
import mujoco_warp as mjw
import warp as wp


@wp.kernel
def get_pose_kernel(
    xpos: wp.array(dtype=wp.vec3, ndim=2),
    xmat: wp.array(dtype=wp.mat33, ndim=2),
    site_ids: wp.array(dtype=int),
    world_id: int,
    out: wp.array(dtype=wp.transform),
):
    i = wp.tid()
    sid = site_ids[i]

    p = xpos[world_id, sid]
    R = xmat[world_id, sid]

    q = wp.quat_from_matrix(R)  # gives xyzw
    out[i] = wp.transform(p, q)


def get_pose(
    data: mjw.Data,
    site_ids: list[int],
    *,
    world_id: int = 0,
    device: str | None = None,
) -> wp.array:
    """Return warp transforms for the given site ids from MJW data."""
    if not site_ids:
        return wp.empty(0, dtype=wp.transform, device=device or data.site_xpos.device)

    if device is None:
        device = data.site_xpos.device

    site_ids_arr = wp.array(site_ids, dtype=int, device=device)
    out = wp.empty(len(site_ids), dtype=wp.transform, device=device)

    wp.launch(
        get_pose_kernel,
        dim=len(site_ids),
        inputs=[data.site_xpos, data.site_xmat, site_ids_arr, int(world_id)],
        outputs=[out],
        device=device,
    )
    return out


if __name__ == "__main__":
    from testing.envs.pipe_insert_2 import PipeInsert2

    env = PipeInsert2()

    mj_model = env.mj_model
    # create mjw.Data object
    mj_data = mj.MjData(mj_model)
    data = mjw.put_data(mj_model, mj_data, nworld=1)

    site_ids = [
        mj_model.site(sn).id for sn in ["keypoint_1", "keypoint_2", "keypoint_3"]
    ]

    T = get_pose(data, site_ids)

    print(T.numpy())
