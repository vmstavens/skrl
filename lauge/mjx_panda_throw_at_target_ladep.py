from typing import Any, Callable, Optional, Tuple

import jax
import mujoco
import numpy as np
from brax.base import System
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from etils import epath
from jax import numpy as jp
from ml_collections import ConfigDict, config_dict
from mujoco_playground._src import mjx_env


def default_config() -> ConfigDict:
    """Returns reward config for the environment."""
    # TODO: update rewards scales with new rewards when defined
    return config_dict.create(
        # Environment timestep. Should match the robot decision frequency.
        dt=0.02,
        # Lowers action magnitude for less-jerky motion.  Also sometimes helps
        # sample efficiency.
        action_scale=0.04,
        # The coefficients for all reward terms used for training.
        reward_scales=config_dict.create(
            # Box goes to the target mocap.
            box_target=12.0,
            # cube has velocity in the direction of the target
            directional_velocity_percentage=3.0,
            # Do not collide the gripper with the floor.
            no_floor_collision=0.25,
            # releasing the cube
            release=1.0,
            # box hits the floor to close to the robot
            box_floor_close=-20.0,
        ),
    )


def _load_sys() -> System:
    """Load a mujoco model from a path."""

    _HERE = epath.Path(__file__).parent.parent
    path = _HERE / "lauge/assets/franka_panda/mjx_single_cube.xml"
    print(path)

    assets = {}
    for f in path.parent.glob("*.xml"):
        assets[f.name] = f.read_bytes()
    for f in (path.parent / "assets").glob("*"):
        assets[f.name] = f.read_bytes()
    xml = path.read_text()
    model = mujoco.MjModel.from_xml_string(xml, assets)
    return mjcf.load_model(model)


def _get_collision_info(
    contact: Any, geom1: int, geom2: int
) -> Tuple[jax.Array, jax.Array]:
    if geom1 > geom2:
        geom1, geom2 = geom2, geom1
    mask = (jp.array([geom1, geom2]) == contact.geom).all(axis=1)
    idx = jp.where(mask, contact.dist, 1e4).argmin()
    dist = contact.dist[idx] * mask[idx]
    normal = (dist < 0) * contact.frame[idx, 0, :3]
    return dist, normal


def _geoms_colliding(state: Optional[State], geom1: int, geom2: int) -> jax.Array:
    return _get_collision_info(state.contact, geom1, geom2)[0] < 0


class PandaThrowAtTarget(PipelineEnv):
    """Environment for training franka panda to bring an object to target."""

    def __init__(self, **kwargs):
        sys = _load_sys()

        self.camera_name = "tracker"

        self._config = config = default_config()

        nsteps = int(np.round(config.dt / sys.opt.timestep))
        kwargs["backend"] = "mjx"
        kwargs["n_frames"] = nsteps
        super().__init__(sys, **kwargs)

        # define constants
        model = sys.mj_model
        arm_joints = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "joint7",
        ]
        finger_joints = ["finger_joint1", "finger_joint2"]
        all_joints = arm_joints + finger_joints
        self._robot_arm_qposadr = np.array(
            [model.jnt_qposadr[model.joint(j).id] for j in arm_joints]
        )
        self._robot_qposadr = np.array(
            [model.jnt_qposadr[model.joint(j).id] for j in all_joints]
        )
        self._gripper_site = model.site("gripper").id
        self._left_finger_geom = model.geom("left_finger_pad").id
        self._right_finger_geom = model.geom("right_finger_pad").id
        self._hand_geom = model.geom("hand_capsule").id
        self._box_geom = model.geom("box").id
        self._box_body = model.body("box").id

        self._box_freejoint_id = model.joint("box").id

        self._box_qposadr = model.jnt_qposadr[model.body("box").jntadr[0]]
        # TODO(btaba): replace with mocap_pos once MJX version 3.2.3 is released.
        self._mocap_id = model.body("mocap_target").mocapid
        self._target_id = model.body("mocap_target").id
        self._floor_geom = model.geom("floor").id
        self._init_q = sys.mj_model.keyframe("home").qpos
        self._init_box_pos = jp.array(
            self._init_q[self._box_qposadr : self._box_qposadr + 3],
            dtype=jp.float32,
        )
        self._init_ctrl = sys.mj_model.keyframe("home").ctrl
        self._lowers = model.actuator_ctrlrange[:, 0]
        self._uppers = model.actuator_ctrlrange[:, 1]

        # load data arrays from keyframe
        data = np.load("lauge/pick_keyframe.npz")
        self._throw_qpos = jp.array(data["qpos"])
        self._throw_qvel = jp.array(data["qvel"])
        self._throw_ctrl = jp.array(data["ctrl"])

    def reset(self, rng: jax.Array) -> State:
        """
        Reset the environment to a state where the robot is holding the box.
        The robot qpos, qvel, and ctrl are loaded from a saved keyframe.
        The target position is still randomized.
        """
        rng, rng_target = jax.random.split(rng)

        # 1. Initialize pipeline state from original reset to get a valid state
        pipeline_state = self.pipeline_init(
            q=self._throw_qpos, qd=self._throw_qvel, ctrl=self._throw_ctrl
        )

        # 3. Randomize target position relative to box position
        box_pos = self._throw_qpos[self._box_qposadr : self._box_qposadr + 3]
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jp.array([-0.2, 1.5, -0.1]),
                maxval=jp.array([0.2, 1.5, 0.1]),
            )
            + box_pos
        )

        # 4. Update target position in pipeline state
        # pipeline_state = pipeline_state.replace(
        #     xpos=pipeline_state.xpos.at[self._target_id, :].set(target_pos)
        # )

        pipeline_state = pipeline_state.replace(
            mocap_pos=pipeline_state.mocap_pos.at[self._mocap_id].set(target_pos)
        )

        # 5. Initialize metrics and info
        metrics = {k: 0.0 for k in self._config.reward_scales.keys()}
        # TODO: redefine out of bounds
        metrics["out_of_bounds"] = 0.0
        # TODO: update info for new task
        self.info = {
            "rng": rng,
            "target_pos": target_pos,
            "released": 0.0,
            "box_last": box_pos,
            "cube_vel": 0.0,
        }

        # 6. Compute initial observation
        obs = self._get_obs(pipeline_state, self.info)

        # 7. Return State object
        return State(
            pipeline_state=pipeline_state,
            obs=obs,
            reward=0.0,
            done=0.0,
            metrics=metrics,
            info=self.info,
        )

    def unit_vector(self, v):
        norm = jp.linalg.norm(v)  # , axis=-1, keepdims=True)
        return jp.where(norm == 0, v, v / norm)

    def step(self, state: State, action: jax.Array) -> State:
        # control update clipped, acts like velocity control
        delta = action * self._config.action_scale
        ctrl = state.pipeline_state.ctrl + delta
        ctrl = jp.clip(ctrl, self._lowers, self._uppers)

        # step the physics
        data = self.pipeline_step(state.pipeline_state, ctrl)

        # compute reward terms
        # use body positions
        target_pos = state.info["target_pos"]
        box_pos = data.xpos[self._box_body]

        # jax.debug.print("here {x}", x=box_pos)

        gripper_pos = data.site_xpos[self._gripper_site]

        # old way of rewarding box distance to target this will probably work in the new version (could be enfore only after release)
        target_direction = target_pos - box_pos
        box_target = 1 - jp.tanh(5 * jp.linalg.norm(target_direction))

        # cube_vel = jp.linalg.norm(box_pos - self.info["box_last"]) / self._config.dt
        # self.info["cube_vel"] = cube_vel
        # self.info["box_last"] = box_pos

        #         data.qvel[dof_address : dof_address + 3]

        cube_vel = data.qvel[self._box_freejoint_id : self._box_freejoint_id + 3]
        print("cube_vel: ", cube_vel)

        unit_direction = self.unit_vector(target_direction)
        unit_velocity = self.unit_vector(cube_vel)
        print(unit_direction, unit_velocity)
        directional_velocity_percentage = jp.dot(unit_direction, unit_velocity)

        # maybe no floor collision check?
        hand_floor_collision = [
            _geoms_colliding(state.pipeline_state, self._floor_geom, g)
            for g in [
                self._left_finger_geom,
                self._right_finger_geom,
                self._hand_geom,
            ]
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = 1 - floor_collision

        # reverse box reached to test for box released
        # state.info["reached_box"] = 1.0 * jp.maximum(
        #     state.info["reached_box"],
        #     (jp.linalg.norm(box_pos - gripper_pos) < 0.012),
        # )

        state.info["released"] = 1.0 * jp.minimum(
            state.info["released"],
            (jp.linalg.norm(box_pos - gripper_pos) > 0.03),
        )

        print(f"{directional_velocity_percentage=}")
        print(f"{state.info["released"]=}")
        release = directional_velocity_percentage * state.info["released"]

        print(f"{release=}")

        box_floor_collision: jax.Array[bool] = _geoms_colliding(
            state.pipeline_state,
            self._floor_geom,
            self._box_geom,
        )
        # if jp.all(box_pos < 1.25):
        #     # if jp.all(box_pos < 1.25):
        #     # if jp.all(box_pos[1] < 1.25):
        #     box_floor_close = box_floor_collision.astype(jp.float32)
        # else:
        #     box_floor_close = 0.0

        # box_floor_close = jp.where(
        #     box_floor_collision,
        #     # box_pos < 1.25,
        # )

        # possible post release movement penalty

        rewards = {
            "box_target": box_target * state.info["released"],
            "directional_velocity_percentage": directional_velocity_percentage,
            "no_floor_collision": no_floor_collision,
            "release": release,
            # "box_floor_close": box_floor_close,
        }
        rewards = {k: v * self._config.reward_scales[k] for k, v in rewards.items()}
        reward = jp.clip(sum(rewards.values()), -1e4, 1e4)

        conditions = jp.array(
            [
                box_pos[0] > 1.0,  # x upper bound
                box_pos[0] < -0.5,  # x lower bound
                box_pos[1] > 1.65,  # y upper bound
                box_pos[1] < -0.5,  # y lower bound
                box_pos[2] > 1.2,  # z upper bound
                box_pos[2] < 0.0,  # z lower bound
                # box_floor_collision,
            ]
        )

        out_of_bounds = jp.any(conditions)
        state.metrics.update(out_of_bounds=out_of_bounds.astype(float), **rewards)

        obs = self._get_obs(data, state.info)
        done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = done.astype(float)
        state = State(data, obs, reward, done, state.metrics, state.info)

        return state

    # TODO: maybe add cube velocity, gripper open?,
    def _get_obs(self, data: State, info: dict[str, Any]) -> jax.Array:
        gripper_pos = data.site_xpos[self._gripper_site]
        gripper_mat = data.site_xmat[self._gripper_site].ravel()
        # obs = jp.concatenate(
        #     [
        #         data.qpos,
        #         data.qvel,
        #         gripper_pos,
        #         gripper_mat[3:],
        #         data.xmat[self._box_body].ravel()[3:],
        #         data.xpos[self._box_body] - data.site_xpos[self._gripper_site],
        #         info["target_pos"] - data.xpos[self._box_body],
        #         data.ctrl - data.qpos[self._robot_qposadr[:-1]],
        #     ]
        # )
        obs = jp.concatenate(
            [
                data.qpos,  # radians → OK
                jp.clip(data.qvel, -10, 10) * 0.5,  # velocities
                gripper_pos,
                gripper_mat[3:],  # already bounded
                data.xmat[self._box_body].ravel()[3:],
                # self.info["cube_vel"],
                jp.clip(
                    data.xpos[self._box_body] - data.site_xpos[self._gripper_site],
                    -10.0,
                    10.0,
                ),
                jp.clip(info["target_pos"] - data.xpos[self._box_body], -10.0, 10.0),
                jp.clip(data.ctrl - data.qpos[self._robot_qposadr[:-1]], -2.0, 2.0),
            ]
        )

        return obs

    def get_demo_video(
        self,
        make_inference_fn: Callable,
        params: tuple,
        n_steps: int = 500,
        render_every: int = 2,
    ) -> np.ndarray:
        jit_reset = jax.jit(self.reset)
        jit_step = jax.jit(self.step)
        inference_fn = make_inference_fn(params, deterministic=True)
        jit_inference_fn = jax.jit(inference_fn)

        # initialize the state
        rng = jax.random.PRNGKey(0)
        state = jit_reset(rng)
        rollout = [state.pipeline_state]

        for i in range(n_steps):
            act_rng, rng = jax.random.split(rng)
            ctrl, _ = jit_inference_fn(state.obs, act_rng)
            state = jit_step(state, ctrl)
            rollout.append(state.pipeline_state)

        # convert to numpy array
        frames = np.array(
            self.render(trajectory=rollout[::render_every], camera=self.camera_name)
        )

        # proper dimensions of frames
        frames = np.transpose(np.array(frames), axes=(0, 3, 1, 2))

        return frames
