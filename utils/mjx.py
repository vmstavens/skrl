from typing import Union

import jaxlie as jaxl
import mujoco as mj
import mujoco.mjx as mjx

from testing.mj import ObjType, does_exist


def get_pose(
    model: mjx.Model, data: mjx.Data, identifier: Union[int, str], obj_type: ObjType
) -> jaxl.SE3:
    """
    Retrieves the pose (position and orientation) of an object in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the object.
    data : mj.MjData
        The simulation data containing the pose information.
    identifier : int or str
        The ID or name of the object.
    obj_type : ObjType
        The type of the object, e.g., body, site, camera.

    Returns
    -------
    sm.SE3
        The pose of the object as an SE3 transformation matrix.

    Raises
    ------
    ValueError
        If the object type does not support pose retrieval.
    """

    if not does_exist(model, identifier, obj_type):
        return jaxl.SE3()

    # Define a mapping for types that can provide poses
    pose_mapping = {
        ObjType.ACTUATOR: data.actuator,
        ObjType.BODY: data.body,
        ObjType.JOINT: data.joint,
        ObjType.GEOM: data.geom,
        ObjType.SITE: data.site,
        ObjType.CAMERA: data.cam,
        ObjType.LIGHT: data.light,
    }

    # Check if type has a pose; raise an error if it does not
    if obj_type not in pose_mapping:
        raise ValueError(f"obj_type {obj_type.name} cannot provide a pose...")

    if obj_type is ObjType.CAMERA:
        xt = data.cam_xpos
        xR = data.cam_xmat
        jaxl.SE3.from_rotation_and_translation()

    if (
        (obj_type is ObjType.CAMERA)
        or (obj_type is ObjType.SITE)
        or (obj_type is ObjType.GEOM)
    ):
        obj = pose_mapping[obj_type](identifier)
        t, q = obj.xpos, smb.r2q(np.reshape(obj.xmat, (3, 3)))
        return make_tf(pos=t, ori=q)

    if obj_type is ObjType.JOINT:
        if JointType(model.joint(identifier).type[0]) is JointType.FREE:
            obj = pose_mapping[obj_type](identifier)
            return make_tf(pos=obj.qpos[0:3], ori=obj.qpos[3:7])

        else:
            obj: mj.mjtJoint = pose_mapping[obj_type](identifier)
            # this is not the x-axis, but the primary axis of the joint. Big difference
            joint_axis = obj.xaxis

            def compute_frame(z_axis: np.ndarray) -> np.ndarray:
                # Normalize the z-axis
                z_axis = z_axis / np.linalg.norm(z_axis)

                # Choose an arbitrary vector not collinear with z_axis
                if np.allclose(z_axis, [1, 0, 0]):
                    v = np.array([0, 1, 0])
                else:
                    v = np.array([1, 0, 0])

                # Compute the x-axis
                x_axis = np.cross(v, z_axis)
                x_axis /= np.linalg.norm(x_axis)

                # Compute the y-axis
                y_axis = np.cross(z_axis, x_axis)

                return np.vstack([x_axis, y_axis, z_axis]).T

            # build rotation matrix from this axis, we choose this to be the z-axis of the rotation matrix
            t, q = obj.xanchor, compute_frame(joint_axis)
            return make_tf(pos=t, ori=q)

    # Retrieve position and quaternion
    obj = pose_mapping[obj_type](identifier)
    t, q = obj.xpos, obj.xquat
    return make_tf(pos=t, ori=q)
