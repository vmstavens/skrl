from pathlib import Path

import glfw
import mujoco as mj
import spatialmath as sm
from dm_control import mjcf

from .base_sim import BaseSim, SimSync, sleep
from .mj import (
    ObjType,
    RobotInfo,
    get_contact_states,
    set_pose,
)


class MjSim(BaseSim):
    def __init__(self):
        super().__init__()

        self._model, self._data = self.init_string()
        # self._model, self._data = self.init()

        self.threads = [self.spin]

    def init_string(self) -> tuple[mj.MjModel, mj.MjData]:
        # <option timestep="0.01"/>
        # <option iterations="1" ls_iterations="5" timestep="0.004" integrator="Euler">
        #     <flag eulerdamp="disable"/>
        # </option>
        _XML = """
        <mujoco>

        <option iterations="1" ls_iterations="5" timestep="0.004" integrator="implicitfast">
            <flag eulerdamp="disable"/>
        </option>

            <worldbody>
                <body name="agent" gravcomp="1">
                    <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
                    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
                    <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
                    <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/>
                </body>
            </worldbody>
            <actuator>
                <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
            </actuator>
        </mujoco>
        """
        #     _XML = """
        #     <mujoco>

        # <option timestep="0.002"
        #     integrator="implicitfast"
        #     solver="Newton"
        #     gravity="0 0 -9.82"
        #     cone="elliptic"
        #     iterations="1"
        #     ls_iterations="1"
        # >
        #     <flag contact="disable" eulerdamp="disable" />
        # </option>

        #         <worldbody>
        #             <body name="agent" gravcomp="1">
        #                 <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
        #                 <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
        #                 <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
        #                 <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0"/>
        #             </body>
        #         </worldbody>
        #         <actuator>
        #             <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
        #             <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
        #             <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
        #         </actuator>
        #     </mujoco>
        #     """
        scene = mj.MjSpec.from_string(_XML)
        m = scene.compile()

        d = mj.MjData(m)
        return m, d

    def init(self) -> tuple[mj.MjModel, mj.MjData]:
        # root
        _HERE = Path(__file__).parent.parent.parent
        # scene path

        self._XML_PATH = Path(__file__).parent / "empty_mjx.xml"

        scene = mj.MjSpec.from_file(self._XML_PATH.as_posix())

        # gripper = mj.MjSpec().from_file(robotiq_2f85_mj_description.MJCF_PATH)

        b_agent = scene.worldbody.add_body(name="agent")
        g_agent = b_agent.add_geom(
            name="agent",
            type=mj.mjtGeom.mjGEOM_BOX,
            size=[0.01, 0.01, 0.01],
            contype=0,
            conaffinity=0,
        )

        b_agent.add_joint(
            name="x", type=mj.mjtJoint.mjJNT_SLIDE, range=[-1, 1], axis=[1, 0, 0]
        )
        b_agent.add_joint(
            name="y", type=mj.mjtJoint.mjJNT_SLIDE, range=[-1, 1], axis=[0, 1, 0]
        )
        b_agent.add_joint(
            name="z", type=mj.mjtJoint.mjJNT_SLIDE, range=[-1, 1], axis=[0, 0, 1]
        )

        scene.add_actuator(
            name="x", trntype=mj.mjtTrn.mjTRN_JOINT, target="x", ctrlrange=[-1, 1]
        ).set_to_position(kp=1000, kv=1000)
        scene.add_actuator(
            name="y", trntype=mj.mjtTrn.mjTRN_JOINT, target="y", ctrlrange=[-1, 1]
        ).set_to_position(kp=1000, kv=1000)
        scene.add_actuator(
            name="z", trntype=mj.mjtTrn.mjTRN_JOINT, target="z", ctrlrange=[-1, 1]
        ).set_to_position(kp=1000, kv=1000)

        m = scene.compile()

        d = mj.MjData(m)
        return m, d

    def spin(self, ss: SimSync):
        while True:
            ss.step()

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def keyboard_callback(self, key: int):
        if key is glfw.KEY_SPACE:
            set_pose(self.model, self.data, "freejoint", ObjType.JOINT, sm.SE3.Tz(1))
            print("You pressed space...")


if __name__ == "__main__":
    sim = MjSim()
    sim.run()
