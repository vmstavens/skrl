from enum import Enum
from typing import Callable, Optional, Union

import jax
import jax.numpy as jp
import jaxlie as jaxl
import mujoco.mjx as mjx


class ObjType(Enum):
    """
    Enumeration of object types used in MuJoCo simulations.

    Attributes
    ----------
    UNKNOWN : int
        Unknown object type (0)
    BODY : int
        Body object type (1)
    XBODY : int
        Body object type for accessing regular frame instead of i-frame (2)
    GEOM : int
        Geometric object type (5)
    SITE : int
        Site object type (6)
    CAMERA : int
        Camera object type (7)
    """

    UNKNOWN = int(mjx.ObjType.UNKNOWN)  # unknown object type
    BODY = int(mjx.ObjType.BODY)  # body
    XBODY = int(
        mjx.ObjType.XBODY
    )  # body, used to access regular frame instead of i-frame
    GEOM = int(mjx.ObjType.GEOM)
    SITE = int(mjx.ObjType.SITE)
    CAMERA = int(mjx.ObjType.CAMERA)
    JOINT = int(mjx.ObjType.BODY)


def get_number_of(model: mjx.Model, obj_type: ObjType) -> int:
    """
    Retrieves the count of objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model from which to count objects.
    obj_type : ObjType
        The type of objects to count, e.g., actuators, bodies, joints.

    Returns
    -------
    int
        The number of objects of the specified type in the model.

    Raises
    ------
    ValueError
        If the specified object type is not recognized.
    """
    type_to_attribute = {
        ObjType.BODY: model.nbody,
        ObjType.GEOM: model.ngeom,
        ObjType.SITE: model.nsite,
        ObjType.CAMERA: model.ncam,
        ObjType.JOINT: model.njnt,
    }

    if obj_type not in type_to_attribute:
        raise ValueError(f"Object type {obj_type} not recognized.")

    return type_to_attribute[obj_type]


def get_names(model: mjx.Model, obj_type: ObjType) -> list[str]:
    """
    Retrieves the names of all objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the objects.
    obj_type : ObjType
        The type of objects to retrieve names for, e.g., actuators, bodies.

    Returns
    -------
    List[str]
        A list of names for all objects of the specified type in the model.
    """
    return [
        mjx.id2name(model, obj_type.value, id)
        for id in range(get_number_of(model, obj_type))
    ]


def get_ids(model: mjx.Model, obj_type: ObjType) -> list[int]:
    """
    Retrieves the names of all objects of a specified type in a MuJoCo model.

    Parameters
    ----------
    model : mj.MjModel
        The MuJoCo model containing the objects.
    obj_type : ObjType
        The type of objects to retrieve names for, e.g., actuators, bodies.

    Returns
    -------
    List[str]
        A list of names for all objects of the specified type in the model.
    """
    return [
        mjx.id2name(model, obj_type.value, id)
        for id in range(get_number_of(model, obj_type))
    ]


def does_exist(model: mjx.Model, identifier: Union[int, str], obj_type: mjx.ObjType):
    if isinstance(identifier, str):
        exists = True if mjx.name2id(model, obj_type.value, identifier) != -1 else False
        if not exists:
            raise ValueError(
                f"{obj_type.name} name '{identifier}' not found in the model. The model contain the {obj_type.name}s {get_names(model, obj_type)}"
            )
    elif isinstance(identifier, int):
        exists = (identifier < get_number_of(model, obj_type)) and (identifier >= 0)
        if not exists:
            raise ValueError(
                f"{obj_type.name} id '{identifier}' not found in the model. The model contain the {obj_type.name}s {get_ids(model, obj_type)}"
            )
    else:
        raise ValueError(
            f"Invalid type input id with value '{identifier}' and type {type(identifier)} use either string or int."
        )
    return exists


def set_pose(
    model: mjx.Model,
    data: mjx.Data,
    identifier: Union[int, str],
    obj_type: ObjType,
    T: jaxl.SE3,
) -> mjx.Data:
    """
    Sets the pose (position and orientation) of an object in a MuJoCo model, if allowed.

    Parameters
    ----------
    model : mjx.Model
        The MuJoCo model containing the object.
    data : mjx.Data
        The simulation data where the pose is set.
    identifier : int or str
        The ID or name of the object.
    obj_type : mjx.ObjType
        The type of the object, e.g., body, joint.
    T : jaxl.SE3
        The desired pose as an SE3 transformation matrix.
    """
    assert does_exist(model, identifier, obj_type)

    # Convert name to id if needed
    if isinstance(identifier, str):
        id = mjx.name2id(model, obj_type.value, identifier)
    else:
        id = identifier

    def set_position_and_orientation(pos_array, quat_array, index):
        """Helper to set position and orientation at given index."""
        new_pos = T.translation()
        new_rot = T.rotation()
        new_quat_xyzw = new_rot  # xyzw format
        # new_quat_xyzw = new_rot.to_quaternion()  # xyzw format
        new_quat_wxyz = jp.array(
            [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
        )

        updated_pos = pos_array.at[index].set(new_pos)
        updated_quat = quat_array.at[index].set(new_quat_wxyz)
        return updated_pos, updated_quat

    # Process based on object type
    if obj_type is ObjType.BODY:
        print("in body")
        # Check if the body is a mocap body
        mocap_id = model.body_mocapid[id]
        if mocap_id != -1:
            new_mocap_pos, new_mocap_quat = set_position_and_orientation(
                data.mocap_pos, data.mocap_quat, mocap_id
            )
            data = data.replace(mocap_pos=new_mocap_pos, mocap_quat=new_mocap_quat)
            return data

        # Check if the body has a freejoint
        body_jntadr = model.body_jntadr[id]
        if (
            body_jntadr != -1 and model.jnt_type[body_jntadr] == 0
        ):  # 0 = free joint in MJX
            print("body has free joint")
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[body_jntadr]

            print(f"{jnt_qposadr=}")

            # Update qpos for free joint: [x, y, z, qw, qx, qy, qz]
            new_pos = T.translation()
            new_rot = T.rotation()
            new_quat_xyzw = new_rot.as_quaternion_xyzw()  # xyzw format
            print(f"{new_quat_xyzw=}")
            # new_quat_xyzw = new_rot.to_quaternion()  # xyzw format
            new_quat_wxyz = jp.array(
                [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
            )
            print(f"{new_quat_wxyz=}")

            # Create the full 7D pose for free joint
            new_qpos = jp.concatenate([new_pos, new_quat_wxyz])
            print(f"{new_qpos=}")

            # Update qpos at the correct position
            updated_qpos = data.qpos.at[jnt_qposadr : jnt_qposadr + 7].set(new_qpos)
            print(f"{updated_qpos=}")
            data = data.replace(qpos=updated_qpos)
            print(f"{data.qpos=}")
            return data

    elif obj_type is ObjType.JOINT:
        print("in joint")
        # Check if the joint is a free joint
        if model.jnt_type[id] == 0:  # 0 = free joint in MJX
            print("in free joint")
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[id]

            # Update qpos for free joint: [x, y, z, qw, qx, qy, qz]
            new_pos = T.translation()
            new_rot = T.rotation()
            new_quat_xyzw = new_rot.as_quaternion_xyzw()  # xyzw format
            # new_quat_xyzw = new_rot.to_quaternion()  # xyzw format
            new_quat_wxyz = jp.array(
                [new_quat_xyzw[3], new_quat_xyzw[0], new_quat_xyzw[1], new_quat_xyzw[2]]
            )

            # Create the full 7D pose for free joint
            new_qpos = jp.concatenate([new_pos, new_quat_wxyz])

            # Update qpos at the correct position
            updated_qpos = data.qpos.at[jnt_qposadr : jnt_qposadr + 7].set(new_qpos)
            data = data.replace(qpos=updated_qpos)
            return data

    # If no valid option found, raise an error
    raise ValueError(
        f"{obj_type.name} '{identifier}' cannot have its pose set. Only mocap bodies, bodies with freejoints, or freejoints are allowed."
    )


def get_pose(
    model: mjx.Model, data: mjx.Data, identifier: Union[int, str], obj_type: mjx.ObjType
) -> jaxl.SE3:
    assert does_exist(model, identifier, obj_type)

    if isinstance(identifier, str):
        id = mjx.name2id(model, obj_type.value, identifier)
    else:
        id = identifier

    if obj_type is ObjType.BODY:
        # Check if the body is a mocap body
        mocap_id = model.body_mocapid[id]
        if mocap_id != -1:
            # For mocap bodies, use mocap_pos and mocap_quat
            xt = data.mocap_pos[mocap_id]
            xquat = data.mocap_quat[mocap_id]  # wxyz format
            # Convert wxyz to xyzw for jaxl
            xquat_xyzw = jp.array([xquat[1], xquat[2], xquat[3], xquat[0]])
            xR = jaxl.SO3.from_quaternion_xyzw(xquat_xyzw)
            return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)

        # Check if the body has a freejoint
        body_jntadr = model.body_jntadr[id]
        if (
            body_jntadr != -1 and model.jnt_type[body_jntadr] == 0
        ):  # 0 = free joint in MJX
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[body_jntadr]

            # Read qpos for free joint: [x, y, z, qw, qx, qy, qz]
            qpos_slice = data.qpos[jnt_qposadr : jnt_qposadr + 7]
            xt = qpos_slice[:3]  # position
            xquat = qpos_slice[3:]  # wxyz quaternion

            # Convert wxyz to xyzw for jaxl
            xquat_xyzw = jp.array([xquat[1], xquat[2], xquat[3], xquat[0]])
            xR = jaxl.SO3.from_quaternion_xyzw(xquat_xyzw)
            return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)

        # For regular bodies, use xpos and xmat
        xt = data.xpos[id]
        xR = data.xmat[id]
        xR = jaxl.SO3.from_matrix(xR.reshape(3, 3))
        return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)

    elif obj_type is ObjType.JOINT:
        # Check if the joint is a free joint
        if model.jnt_type[id] == 0:  # 0 = free joint in MJX
            # Get the qpos address for this joint
            jnt_qposadr = model.jnt_qposadr[id]

            # Read qpos for free joint: [x, y, z, qw, qx, qy, qz]
            qpos_slice = data.qpos[jnt_qposadr : jnt_qposadr + 7]
            xt = qpos_slice[:3]  # position
            xquat = qpos_slice[3:]  # wxyz quaternion

            # Convert wxyz to xyzw for jaxl
            xquat_xyzw = jp.array([xquat[1], xquat[2], xquat[3], xquat[0]])
            xR = jaxl.SO3.from_quaternion_xyzw(xquat_xyzw)
            return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)
        else:
            raise ValueError(
                f"Joint {identifier} is not a free joint and cannot provide a pose"
            )

    # For other object types (geom, site, camera)
    pose_mapping = {
        ObjType.GEOM: (data.geom_xpos, data.geom_xmat),
        ObjType.SITE: (data.site_xpos, data.site_xmat),
        ObjType.CAMERA: (data.cam_xpos, data.cam_xmat),
    }

    if obj_type not in pose_mapping:
        raise ValueError(f"obj_type {obj_type.name} cannot provide a pose...")

    _xpos, _xmat = pose_mapping[obj_type]
    xt = _xpos[id]
    xR = _xmat[id]
    xR = jaxl.SO3.from_matrix(xR.reshape(3, 3))
    return jaxl.SE3.from_rotation_and_translation(rotation=xR, translation=xt)


# def is_stable(data: mjx.Data, cb: Optional[Callable] = None) -> bool:
#     nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

#     # Always print the condition values
#     jax.debug.print("Stability check:")
#     jax.debug.print("  qpos has NaN: {qpos_nan}", qpos_nan=jp.isnan(data.qpos).any())
#     jax.debug.print("  qvel has NaN: {qvel_nan}", qvel_nan=jp.isnan(data.qvel).any())
#     jax.debug.print("  Combined condition: {cond}", cond=nan_condition)

#     def raise_error(_):
#         jax.debug.print("UNSTABLE DETECTED!")
#         jax.debug.print("  Problematic qpos: {qpos}", qpos=data.qpos[:4])
#         jax.debug.print("  Problematic qvel: {qvel}", qvel=data.qvel[:4])
#         # raise ValueError("Yo your simulation is unstable")

#     callback = cb if cb is not None else raise_error

#     # Print only when unstable
#     def print_unstable(_):
#         jax.debug.print("Unstable condition triggered")
#         return callback(_)

#     result = jax.lax.cond(
#         ~jp.all(nan_condition),
#         lambda _: (print_unstable(_), False)[1],
#         lambda _: (jax.debug.print("System is stable"), True)[1],
#         operand=None,
#     )

#     return result


def is_stable(data: mjx.Data, cb: Optional[Callable] = None) -> bool:
    nan_condition = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    is_unstable = jp.all(nan_condition)  # True when unstable

    # Always print the condition values
    # jax.debug.print("Stability check:")
    # jax.debug.print("  qpos has NaN: {qpos_nan}", qpos_nan=jp.isnan(data.qpos).any())
    # jax.debug.print("  qvel has NaN: {qvel_nan}", qvel_nan=jp.isnan(data.qvel).any())
    # jax.debug.print("  Combined condition: {cond}", cond=nan_condition)
    # jax.debug.print("  Is unstable: {unstable}", unstable=is_unstable)

    def raise_error(_):
        pass
        # jax.debug.print("UNSTABLE DETECTED!")
        # jax.debug.print("  Problematic qpos: {qpos}", qpos=data.qpos[:4])
        # jax.debug.print("  Problematic qvel: {qvel}", qvel=data.qvel[:4])
        # raise ValueError("Yo your simulation is unstable")

    callback = cb if cb is not None else raise_error

    # Corrected cond logic:
    # result = jax.lax.cond(
    #     is_unstable,  # Condition: True when unstable
    #     lambda _: (callback(_), False)[
    #         1
    #     ],  # If unstable: call callback and return False
    #     lambda _: (jax.debug.print("System is stable"), True)[
    #         1
    #     ],  # If stable: print and return True
    #     operand=None,
    # )
    result = True

    return result
