from pathlib import Path

import mujoco as mj
import numpy as np
from robot_descriptions import robotiq_2f85_mj_description


def init() -> mj.MjModel:
    # root
    _HERE = Path(__file__).parent.parent.parent
    # scene path

    _empty_scene = """
<mujoco model="empty scene">

<compiler angle="radian" autolimits="true" />
<option timestep="0.002"
    integrator="implicitfast"
    solver="Newton"
    gravity="0 0 -9.82"
    cone="elliptic"
    sdf_iterations="5"
    sdf_initpoints="30"
    noslip_iterations="2"
>
    <!-- impratio="100" -->
    <!-- mjMAXCONPAIR="10" -->
    <flag multiccd="enable" nativeccd="enable" />
    <!-- <flag nativeccd="enable" /> -->
</option>

<custom>
    <numeric data="15" name="max_contact_points" />
    <numeric data="15" name="max_geom_pairs" />
</custom>

<extension>
    <plugin plugin="mujoco.sensor.touch_grid" />
    <!-- <plugin plugin="mujoco.elasticity.solid" /> -->
    <!-- <plugin plugin="mujoco.elasticity.shell" /> -->
</extension>

<statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

<visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
    <rgba haze="0.15 0.25 0.35 1" />
    <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />
    <!-- <global azimuth="120" elevation="-20" offwidth="1920" offheight="1080" /> -->

</visual>

<asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512"
        height="3072" />
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
        rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300" />
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"
        reflectance="0.2" />
</asset>
<worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" />
    <geom name="floor" size="0 0 0.5" type="plane" material="groundplane"
        solimp="0.0 0.0 0.0 0.0 1" />
    <!-- <geom name="floor" size="0 0 0.5" type="plane" material="groundplane" /> -->

</worldbody>
</mujoco>
"""

    scene = mj.MjSpec.from_string(_empty_scene)

    gripper = mj.MjSpec().from_file(robotiq_2f85_mj_description.MJCF_PATH)

    gripper.worldbody.first_body().add_site(name="tcp", pos=[0, 0, 0.15])

    gripper.worldbody.first_body().add_joint(
        name="x", type=mj.mjtJoint.mjJNT_SLIDE, range=[-1, 1], axis=[1, 0, 0]
    )
    gripper.worldbody.first_body().add_joint(
        name="y", type=mj.mjtJoint.mjJNT_SLIDE, range=[-1, 1], axis=[0, 1, 0]
    )
    gripper.worldbody.first_body().add_joint(
        name="z", type=mj.mjtJoint.mjJNT_SLIDE, range=[-1, 1], axis=[0, 0, 1]
    )

    gripper.add_actuator(
        name="x", trntype=mj.mjtTrn.mjTRN_JOINT, target="x", ctrlrange=[-1, 1]
    ).set_to_position(kp=1000, kv=1000)
    gripper.add_actuator(
        name="y", trntype=mj.mjtTrn.mjTRN_JOINT, target="y", ctrlrange=[-1, 1]
    ).set_to_position(kp=1000, kv=1000)
    gripper.add_actuator(
        name="z", trntype=mj.mjtTrn.mjTRN_JOINT, target="z", ctrlrange=[-1, 1]
    ).set_to_position(kp=1000, kv=1000)

    joint = scene.worldbody.add_frame(
        name="cable",
        pos=[0, 0, 0],
        euler=np.array([0, 0, 0]),
        # euler=np.array([np.pi, 0, 0]),
    ).attach_body(gripper.worldbody.first_body())

    m = scene.compile()
    return m


def _post_init(model: mj.MjModel, data: mj.MjData) -> None:
    pos_joint_ids = [
        model.joint("x").id,
        model.joint("y").id,
        model.joint("z").id,
    ]
    pos_ctrl_ids = [
        model.actuator("x").id,
        model.actuator("y").id,
        model.actuator("z").id,
    ]
    # get the slide joint addresses, we can use these by data.qpos.at[self._free_qposaddr].set([new_qpos_segment])
    free_qposaddr = model.jnt_qposadr[pos_joint_ids]
    # get the slide actuator addresses, we can use these by data.ctrl.at[self._free_ctrladdr].set([new_ctrl_segment])
    free_ctrladdr = pos_ctrl_ids
    # get control limits
    ctrl_min = model.actuator_ctrlrange[1:, 0]
    ctrl_max = model.actuator_ctrlrange[1:, 1]
    # get joint limits
    qpos_min = model.jnt_range[free_qposaddr, 0]
    qpos_max = model.jnt_range[free_qposaddr, 1]


if __name__ == "__main__":
    m = init()
    d = mj.MjData(m)

    _post_init(m, d)
