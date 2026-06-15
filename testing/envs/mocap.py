import functools
import math
import os
from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco as mj
import numpy as np
import tyro
from brax.envs.wrappers import training as brax_training
from brax.training import acting
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env, wrapper
from robot_descriptions import robotiq_2f85_mj_description


def empty(floor: bool = False) -> mj.MjSpec:

    # integrator="RK4"
    _XML = f"""
        <mujoco model="empty scene">

        <compiler angle="radian" autolimits="true" />
        <option 
            timestep="0.0001" 
            integrator="RK4" 
            solver="Newton" 
            gravity="0 0 -9.82" 
            cone="elliptic" 
        >
            <flag multiccd="enable" />
        </option>

        <statistic center="0.3 0 0.3" extent="0.8" meansize="0.08" />

        <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0 0 0" />
            <rgba haze="0.15 0.25 0.35 1" />
            <global azimuth="120" elevation="-20" offwidth="2000" offheight="2000" />
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
            {'<geom name="floor" size="0 0 0.5" type="plane" material="groundplane" />' if floor else ""}
        </worldbody>

    </mujoco>
    """
    return mj.MjSpec.from_string(_XML)


def _ensure_warp_internal_module() -> None:
    """Ensure warp._src is available on the warp package."""
    try:
        import importlib

        import warp as wp

        try:
            getattr(wp, "_src")
            return
        except AttributeError:
            pass

        wp_src = importlib.import_module("warp._src")
        setattr(wp, "_src", wp_src)
    except Exception:
        return


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=150,
        action_repeat=1,
        action_scale_pos=0.02,
        action_scale_rot=0.2,
        grip_weight=0.25,
        success_threshold_item=0.03,
        grip_ctrl_min=0.0,
        grip_ctrl_max=255.0,
        # impl="jax",
        impl="warp",
    )


def ___build_model() -> mj.MjModel:
    xml = """
<mujoco model="mocap_reach">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="effector" pos="0 0 0">
      <joint name="free" type="free"/>
      <geom type="sphere" size="0.02" rgba="0.2 0.6 0.9 1"/>
      <site name="ee" size="0.01" rgba="0.2 0.6 0.9 1"/>
    </body>

    <body name="ee_mocap" mocap="true" pos="0 0 0">
      <site name="ee_mocap_site" size="0.008" rgba="0 1 0 0.3"/>
    </body>

    <body name="goal_mocap" mocap="true" pos="0.2 0.1 0.0">
      <site name="goal" size="0.01" rgba="1 0 0 1"/>
    </body>
  </worldbody>

  <equality>
    <weld name="ee_weld" body1="effector" body2="ee_mocap" solref="0.01 1"/>
  </equality>
</mujoco>
    """
    spec = mj.MjSpec.from_string(xml)
    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
    ee_mocap_body = spec.body("ee_mocap")
    ee_mocap_body.add_frame(name="gripper_mount").attach_body(
        gripper.worldbody.first_body(),
        prefix="gripper/",
    )
    return spec.compile()


def __build_model() -> mj.MjModel:

    scene = empty(floor=False)
    # 1) goal
    b_goal = scene.worldbody.add_body(name="goal", mocap=True)
    b_goal.add_geom(
        name="goal",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
        rgba=[1, 0, 0, 1],
    )
    s_goal = b_goal.add_site(name="goal")

    # 2) mocap
    b_mocap = scene.worldbody.add_body(name="mocap", mocap=True)
    b_mocap.add_geom(
        name="mocap",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
    )
    s_mocap = b_mocap.add_site(name="mocap")

    # 3) gripper (welded to mocap site)
    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
    b_gripper = gripper.worldbody.first_body()
    b_gripper.add_freejoint()
    b_gripper.add_site(name="gripper")
    f_gripper = scene.worldbody.add_frame(name="gripper")
    f_gripper.attach_body(b_gripper, prefix="gripper/")

    scene.add_equality(
        name="weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="gripper/gripper",
        name2="mocap",
        # omit data => MuJoCo uses the current relative pose as the target
        solref=[0.000001, 2],
    )

    return scene.compile()


def _build_model() -> mj.MjModel:

    scene = empty(floor=True)
    # 1) goal
    b_goal = scene.worldbody.add_body(name="goal", mocap=True, pos=[0.1, 0.1, 0.1])
    b_goal.add_geom(
        name="goal",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
        rgba=[1, 0, 0, 1],
    )
    s_goal = b_goal.add_site(name="goal")

    # 2) mocap
    b_mocap = scene.worldbody.add_body(
        name="mocap", mocap=True, pos=[-0.1, -0.1, 0.5], euler=[3.14, 0, 0]
    )
    b_mocap.add_geom(
        name="mocap",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
    )
    s_mocap = b_mocap.add_site(name="mocap")

    # 3) gripper (welded to mocap site)
    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)
    b_gripper = gripper.worldbody.first_body()
    b_gripper.add_freejoint()
    b_gripper.add_site(name="gripper")
    f_gripper = scene.worldbody.add_frame(
        name="gripper", pos=[-0.1, -0.1, 0.5], euler=[3.14, 0, 0]
    )
    f_gripper.attach_body(b_gripper, prefix="gripper/")

    b_item = scene.worldbody.add_body(name="item", pos=[0, 0, 0.02])
    b_item.add_geom(
        name="item",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        rgba=[0, 1, 0, 1],
    )
    b_item.add_site(name="item")
    b_item.add_freejoint()

    scene.add_equality(
        name="weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="gripper/gripper",
        name2="mocap",
        # omit data => MuJoCo uses the current relative pose as the target
        solref=[0.000001, 2],
    )

    return scene.compile()


def _quat_conj(q: jax.Array) -> jax.Array:
    q = jp.asarray(q)
    return jp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def _quat_mul(q1: jax.Array, q2: jax.Array) -> jax.Array:
    q1 = jp.asarray(q1)
    q2 = jp.asarray(q2)
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return jp.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=-1,
    )


def _quat_normalize(q: jax.Array, eps: float = 1e-8) -> jax.Array:
    q = jp.asarray(q)
    return q / (jp.linalg.norm(q, axis=-1, keepdims=True) + eps)


def _quat_to_mat(q: jax.Array) -> jax.Array:
    q = _quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m00 = 1.0 - 2.0 * (y * y + z * z)
    m01 = 2.0 * (x * y - z * w)
    m02 = 2.0 * (x * z + y * w)
    m10 = 2.0 * (x * y + z * w)
    m11 = 1.0 - 2.0 * (x * x + z * z)
    m12 = 2.0 * (y * z - x * w)
    m20 = 2.0 * (x * z - y * w)
    m21 = 2.0 * (y * z + x * w)
    m22 = 1.0 - 2.0 * (x * x + y * y)
    return jp.stack(
        [
            jp.stack([m00, m01, m02], axis=-1),
            jp.stack([m10, m11, m12], axis=-1),
            jp.stack([m20, m21, m22], axis=-1),
        ],
        axis=-2,
    )


def _rotvec_from_mat(mat: jax.Array, eps: float = 1e-6) -> jax.Array:
    trace = mat[..., 0, 0] + mat[..., 1, 1] + mat[..., 2, 2]
    cos_theta = jp.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    theta = jp.arccos(cos_theta)
    sin_theta = jp.sin(theta)
    vee = jp.stack(
        [
            mat[..., 2, 1] - mat[..., 1, 2],
            mat[..., 0, 2] - mat[..., 2, 0],
            mat[..., 1, 0] - mat[..., 0, 1],
        ],
        axis=-1,
    )
    axis = vee / (2.0 * sin_theta[..., None] + eps)
    rotvec = axis * theta[..., None]
    rotvec_small = 0.5 * vee
    return jp.where(theta[..., None] < eps, rotvec_small, rotvec)


def _rotvec_to_quat(v: jax.Array, eps: float = 1e-6) -> jax.Array:
    v = jp.asarray(v)
    angle = jp.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * angle
    sin_half = jp.sin(half)
    axis = jp.where(angle > eps, v / angle, jp.zeros_like(v))
    quat = jp.concatenate([jp.cos(half), axis * sin_half], axis=-1)
    quat_small = jp.concatenate([jp.ones_like(half), 0.5 * v], axis=-1)
    quat = jp.where(angle > eps, quat, quat_small)
    return _quat_normalize(quat)


def _quat_to_rotvec(q: jax.Array, eps: float = 1e-6) -> jax.Array:
    q = _quat_normalize(q)
    q = jp.where(q[..., 0:1] < 0.0, -q, q)
    w = jp.clip(q[..., 0], -1.0, 1.0)
    v = q[..., 1:]
    sin_half = jp.linalg.norm(v, axis=-1)
    angle = 2.0 * jp.arctan2(sin_half, w)
    scale = jp.where(sin_half > eps, angle / sin_half, 2.0)
    return v * scale[..., None]


def _random_quat(rng: jax.Array) -> jax.Array:
    rng_axis, rng_angle = jax.random.split(rng)
    axis = jax.random.normal(rng_axis, (3,))
    axis = axis / (jp.linalg.norm(axis) + 1e-8)
    angle = jax.random.uniform(rng_angle, (), minval=-jp.pi, maxval=jp.pi)
    return _rotvec_to_quat(axis * angle)


def _as_index(value) -> int:
    return int(np.asarray(value).item())


def _metric_to_float(value) -> float:
    if value is None:
        return math.nan
    try:
        return float(np.asarray(value))
    except (TypeError, ValueError):
        return math.nan


def _plot_metrics(history: dict[str, list[float]], path: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    steps = history.get("step", [])
    if not steps:
        return

    plot_specs = [
        ("eval/episode_reward", "Eval episode reward"),
        ("training/total_loss", "Training total loss"),
        ("training/sps", "Training steps/sec"),
    ]

    fig, axes = plt.subplots(len(plot_specs), 1, figsize=(7, 8), sharex=True)
    if len(plot_specs) == 1:
        axes = [axes]

    for ax, (key, title) in zip(axes, plot_specs):
        values = history.get(key)
        if not values:
            continue
        ax.plot(steps, values, marker="o", linewidth=1.2)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("training step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _save_frames(frames: list[np.ndarray], path: str, fps: int = 30) -> None:
    if not frames:
        return
    array = np.stack(frames, axis=0)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, array, fps=fps)
        return
    except Exception:
        pass

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not path.endswith(".png"):
            path = f"{path}.png"
        plt.imsave(path, array[0])
        return
    except Exception:
        pass

    if not path.endswith(".npy"):
        path = f"{path}.npy"
    np.save(path, array)


def _format_eval_path(base_path: str, eval_index: int, step: int) -> str:
    if not base_path:
        return ""
    if "{" in base_path and "}" in base_path:
        try:
            return base_path.format(step=step, eval=eval_index)
        except (KeyError, ValueError):
            pass
    stem, ext = os.path.splitext(base_path)
    suffix = f"_eval{eval_index:03d}_step{step}"
    return f"{stem}{suffix}{ext}"


def render_rollout(
    env: mjx_env.MjxEnv,
    steps: int,
    path: str,
    seed: int = 0,
    fps: int = 30,
    policy=None,
) -> None:
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)
    trajectory = [state]
    if policy is None:
        action = jp.zeros((env.action_size,), dtype=jp.float32)
        for _ in range(steps):
            state = env.step(state, action)
            trajectory.append(state)
    else:
        for _ in range(steps):
            rng, step_key = jax.random.split(rng)
            action, _ = policy(state.obs, step_key)
            state = env.step(state, action)
            trajectory.append(state)
    frames = env.render(trajectory, height=480, width=640)
    _save_frames(frames, path, fps=fps)


class MocapReach(mjx_env.MjxEnv):
    def __init__(self, config: config_dict.ConfigDict = default_config()):
        super().__init__(config)
        if config.impl not in {"jax", "warp", "c"}:
            raise ValueError(
                f"Invalid MJX impl '{config.impl}' (expected: jax, warp, c)"
            )
        if config.impl == "warp":
            _ensure_warp_internal_module()
        self._mj_model = _build_model()
        self._mj_model.opt.timestep = float(config.sim_dt)
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)
        self._episode_length = config.episode_length
        self._action_scale_pos = config.action_scale_pos
        self._action_scale_rot = config.action_scale_rot
        self._grip_weight = config.grip_weight
        self._success_threshold_item = config.success_threshold_item
        self._grip_ctrl_min = float(config.grip_ctrl_min)
        self._grip_ctrl_max = float(config.grip_ctrl_max)

        self._ee_site_id = _as_index(self._mj_model.site("gripper/pinch").id)
        self._goal_site_id = _as_index(self._mj_model.site("goal").id)
        self._item_site_id = _as_index(self._mj_model.site("item").id)
        self._ee_mocap_id = _as_index(self._mj_model.body("mocap").mocapid)
        self._goal_mocap_id = _as_index(self._mj_model.body("goal").mocapid)
        self._mocap_body_id = _as_index(self._mj_model.body("mocap").id)

        self._goal_low = jp.array([-0.3, -0.3, 0.2])
        self._goal_high = jp.array([0.3, 0.3, 0.4])

        self._item_low = jp.array([-0.3, -0.3, 0.02])
        self._item_high = jp.array([0.3, 0.3, 0.02])

        self._workspace_low = jp.minimum(self._goal_low, self._item_low)
        self._workspace_high = jp.maximum(self._goal_high, self._item_high)

        self._mocap_home_pos = jp.array(self._mj_model.body_pos[self._mocap_body_id])
        self._mocap_home_quat = jp.array(self._mj_model.body_quat[self._mocap_body_id])

        self._gripper_body_id = _as_index(self._mj_model.body("gripper/base_mount").id)
        self._item_body_id = _as_index(self._mj_model.body("item").id)
        self._gripper_qpos_adr = _as_index(
            self._mj_model.jnt_qposadr[
                self._mj_model.body_jntadr[self._gripper_body_id]
            ]
        )
        self._item_qpos_adr = _as_index(
            self._mj_model.jnt_qposadr[self._mj_model.body_jntadr[self._item_body_id]]
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_goal, rng_item = jax.random.split(rng, 3)
        goal = jax.random.uniform(
            rng_goal, shape=(3,), minval=self._goal_low, maxval=self._goal_high
        )
        item_pos = jax.random.uniform(
            rng_item, shape=(3,), minval=self._item_low, maxval=self._item_high
        )
        mocap_pos = jp.clip(
            self._mocap_home_pos, self._workspace_low, self._workspace_high
        )
        mocap_quat = self._mocap_home_quat

        data = mjx.make_data(
            self._mj_model,
            impl=self._mjx_model.impl.value,
            naconmax=1500,
            # nconmax=1500,
            njmax=500,
            # naconmax=50,
        )

        item_pose = jp.concatenate([item_pos, jp.array([1.0, 0.0, 0.0, 0.0])])
        gripper_pose = jp.concatenate([mocap_pos, mocap_quat])
        qpos = data.qpos
        qpos = qpos.at[self._item_qpos_adr : self._item_qpos_adr + 7].set(item_pose)
        qpos = qpos.at[self._gripper_qpos_adr : self._gripper_qpos_adr + 7].set(
            gripper_pose
        )
        data = data.replace(qpos=qpos)

        # set robot pose
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._ee_mocap_id].set(mocap_pos),
            mocap_quat=data.mocap_quat.at[self._ee_mocap_id].set(mocap_quat),
        )

        # set goal pose
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocap_id].set(goal),
            mocap_quat=data.mocap_quat.at[self._goal_mocap_id].set(
                jp.array([1.0, 0.0, 0.0, 0.0])
            ),
        )

        data = data.replace(
            ctrl=data.ctrl.at[0].set(self._grip_ctrl_min),
        )

        data = mjx.forward(self._mjx_model, data)

        obs = self._get_obs(data)
        info = {"rng": rng, "step": jp.array(0)}
        reward = jp.array(0.0)
        done = jp.array(0.0)
        item_err = jp.linalg.norm(obs[:3])
        grip_err = jp.linalg.norm(obs[3:6])
        metrics = {
            "reward": reward,
            "item_error": item_err,
            "grip_error": grip_err,
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        dpos = jp.clip(action[:3], -1.0, 1.0) * self._action_scale_pos
        drot = jp.clip(action[3:], -1.0, 1.0) * self._action_scale_rot
        grip_action = jp.clip(action[6], -1.0, 1.0)
        grip_ctrl = (grip_action + 1.0) * 0.5 * (
            self._grip_ctrl_max - self._grip_ctrl_min
        ) + self._grip_ctrl_min
        target = state.data.mocap_pos[self._ee_mocap_id] + dpos
        target = jp.clip(target, self._workspace_low, self._workspace_high)
        delta_quat = _rotvec_to_quat(drot)
        target_quat = _quat_mul(delta_quat, state.data.mocap_quat[self._ee_mocap_id])
        target_quat = _quat_normalize(target_quat)

        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[self._ee_mocap_id].set(target),
            mocap_quat=state.data.mocap_quat.at[self._ee_mocap_id].set(target_quat),
            ctrl=state.data.ctrl.at[0].set(grip_ctrl),
        )

        data = mjx_env.step(self._mjx_model, data, data.ctrl, self.n_substeps)

        obs = self._get_obs(data)
        item_err = jp.linalg.norm(obs[:3])
        grip_err = jp.linalg.norm(obs[3:6])
        reward = -(item_err + self._grip_weight * grip_err)
        done = item_err < self._success_threshold_item
        timeout = state.info["step"] >= self._episode_length
        done = done | timeout

        info = {**state.info, "step": state.info["step"] + 1}
        metrics = {
            **state.metrics,
            "reward": reward,
            "item_error": item_err,
            "grip_error": grip_err,
        }
        return mjx_env.State(data, obs, reward, done.astype(float), metrics, info)

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        item_pos = data.site_xpos[self._item_site_id]
        goal_pos = data.site_xpos[self._goal_site_id]
        ee_pos = data.site_xpos[self._ee_site_id]
        item_err = goal_pos - item_pos
        grip_err = item_pos - ee_pos
        denom = self._grip_ctrl_max - self._grip_ctrl_min
        ctrl_norm = jp.where(
            denom > 0.0,
            (data.ctrl[0] - self._grip_ctrl_min) / denom * 2.0 - 1.0,
            0.0,
        )
        return jp.concatenate([item_err, grip_err, jp.array([ctrl_norm])])

    @property
    def action_size(self) -> int:
        return 7

    @property
    def observation_size(self) -> int:
        return 7

    @property
    def xml_path(self) -> str:
        return "mocap_reach.xml"

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


class SafeAutoResetWrapper(wrapper.Wrapper):
    """Auto-reset wrapper that tolerates non-batched mjx.Data fields."""

    def reset(self, rng: jax.Array) -> mjx_env.State:
        state = self.env.reset(rng)
        state.info["first_state"] = state.data
        state.info["first_obs"] = state.obs
        return state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        if "steps" in state.info:
            steps = state.info["steps"]
            steps = jp.where(state.done, jp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))
            return jp.where(done, x, y)

        data = jax.tree.map(where_done, state.info["first_state"], state.data)
        obs = jax.tree.map(where_done, state.info["first_obs"], state.obs)
        return state.replace(data=data, obs=obs)


def wrap_for_mjx_training(
    env: mjx_env.MjxEnv,
    episode_length: int,
    action_repeat: int,
    randomization_fn=None,
) -> wrapper.Wrapper:
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = SafeAutoResetWrapper(env)
    return env


@dataclass
class TrainArgs:
    num_timesteps: int = 200_000
    num_evals: int = 10
    episode_length: int = 1000
    action_repeat: int = 1
    unroll_length: int = 10
    num_minibatches: int = 16
    num_updates_per_batch: int = 4
    discounting: float = 0.99
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    num_envs: int = 256
    # num_envs: int = 256
    batch_size: int = 4096
    seed: int = 0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    deterministic_eval: bool = False
    metrics_plot_path: str = "testing/envs/mocap_metrics_pose_2f85_bring.png"
    render_only: bool = False
    render_path: str = "testing/envs/mocap_rollout.gif"
    render_steps: int = 1000
    render_fps: int = 30
    render_eval_rollouts: bool = True
    render_eval_path: str = ""


def main() -> None:
    args = tyro.cli(TrainArgs)

    print("config...")

    cfg = default_config()
    cfg.episode_length = args.episode_length
    cfg.action_repeat = args.action_repeat

    print("env...")
    env = MocapReach(cfg)

    if args.render_only:
        render_rollout(
            env,
            steps=args.render_steps,
            path=args.render_path,
            seed=args.seed,
            fps=args.render_fps,
        )
        print(f"saved rollout to {args.render_path}")
        return

    plot_keys = [
        "eval/episode_reward",
        "training/total_loss",
        "training/sps",
    ]
    metrics_history: dict[str, list[float]] = {key: [] for key in plot_keys}
    metrics_history["step"] = []
    eval_rollout_index = 0

    def progress_fn(step: int, metrics: dict) -> None:
        print(metrics)
        metrics_history["step"].append(step)
        for key in plot_keys:
            metrics_history[key].append(_metric_to_float(metrics.get(key)))
        _plot_metrics(metrics_history, args.metrics_plot_path)

    def policy_params_fn(step: int, make_policy, params) -> None:
        nonlocal eval_rollout_index
        if not args.render_eval_rollouts:
            return
        if jax.process_index() != 0:
            return
        base_path = args.render_eval_path or args.render_path
        if not base_path:
            return
        eval_path = _format_eval_path(base_path, eval_rollout_index, int(step))
        policy = make_policy(params, deterministic=args.deterministic_eval)
        seed = args.seed + eval_rollout_index
        eval_rollout_index += 1
        render_rollout(
            env,
            steps=args.render_steps,
            path=eval_path,
            seed=seed,
            fps=args.render_fps,
            policy=policy,
        )
        print(f"saved eval rollout to {eval_path}")

    print("train func def...")
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.num_timesteps,
        num_evals=args.num_evals,
        reward_scaling=args.reward_scaling,
        episode_length=args.episode_length,
        normalize_observations=args.normalize_observations,
        action_repeat=args.action_repeat,
        unroll_length=args.unroll_length,
        num_minibatches=args.num_minibatches,
        num_updates_per_batch=args.num_updates_per_batch,
        discounting=args.discounting,
        learning_rate=args.learning_rate,
        entropy_cost=args.entropy_cost,
        num_envs=args.num_envs,
        batch_size=args.batch_size,
        seed=args.seed,
        gae_lambda=args.gae_lambda,
        clipping_epsilon=args.clipping_epsilon,
        deterministic_eval=args.deterministic_eval,
        network_factory=ppo_networks.make_ppo_networks,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
    )

    print("train...")

    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrap_for_mjx_training,
    )

    print("training metrics:", metrics)

    if args.render_path:
        render_rollout(
            env,
            steps=args.render_steps,
            path=args.render_path,
            seed=args.seed,
            fps=args.render_fps,
        )
        print(f"saved rollout to {args.render_path}")

    # quick deterministic rollout after training
    policy = make_inference_fn(params)
    rng = jax.random.PRNGKey(args.seed)
    eval_env = wrap_for_mjx_training(
        env, episode_length=args.episode_length, action_repeat=args.action_repeat
    )
    rng, reset_key = jax.random.split(rng)
    reset_keys = jax.random.split(reset_key, 1)
    state = eval_env.reset(reset_keys)
    for _ in range(args.episode_length):
        rng, step_key = jax.random.split(rng)
        state, _ = acting.actor_step(eval_env, state, policy, step_key)
    final_obs = np.asarray(state.obs[0])
    final_item_err = np.linalg.norm(final_obs[:3])
    final_grip_err = np.linalg.norm(final_obs[3:6])
    print(
        "final errors (unbatched): "
        f"item={float(final_item_err):.4f}, grip={float(final_grip_err):.4f}"
    )


if __name__ == "__main__":
    main()
