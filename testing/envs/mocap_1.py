import functools
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Union

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


def _parse_float_list(value: Optional[str]) -> list[float]:
    if not value:
        return []
    return [float(item) for item in value.strip().split()]


def _parse_str_list(value: Optional[float]) -> list[str]:
    if not value:
        return []
    return "".join([f"{x} " for x in value])


def mjx_cable(
    *,
    model_name: str = "cable",
    prefix: str = "cable:",
    curve: str = "0 s 0",
    count: Union[str, int] = "10 1 1",
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    size: Union[str, float, int] = 1,
    segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: Union[str, list[float]] = "0.2 0.2 0.2 1",
    initial: str = "free",
    solref: Optional[list] = None,
) -> mj.MjSpec:
    del model_name, curve, vmax
    base_pos = [0.0, 0.0, 0.0]
    damping = 1e-2
    armature = 0.001
    friction = [0.7, 0.7, 0.7]
    if solref is None:
        solref = [0.000001, 1.0]
    # solref = [0.001, 3.0]
    condim = 4
    geom_type = "capsule"
    add_freejoint = initial == "free"
    if isinstance(count, str):
        count_tokens = count.strip().split()
        n_segments = int(count_tokens[0]) if count_tokens else 0
    else:
        n_segments = int(count)
    size_value = float(size.split()[0]) if isinstance(size, str) else float(size)
    segment_length = size_value / max(n_segments, 1)
    radius = segment_size
    rgba = _parse_float_list(rgba) if isinstance(rgba, str) else rgba
    if len(rgba) != 4:
        rgba = [0.2, 0.2, 0.2, 1.0]
    name_prefix = prefix[:-1] if prefix.endswith(":") else prefix
    geom_type_map = {
        "cylinder": mj.mjtGeom.mjGEOM_CYLINDER,
        "capsule": mj.mjtGeom.mjGEOM_CAPSULE,
        "sphere": mj.mjtGeom.mjGEOM_SPHERE,
        "box": mj.mjtGeom.mjGEOM_BOX,
    }
    if geom_type not in geom_type_map:
        raise ValueError(f"Unsupported geom_type: {geom_type}")

    if n_segments <= 0:
        raise ValueError("n_segments must be > 0")

    # Cable runs along +Y in the local frame, matching curve="0 s 0".
    geom_euler = [-np.pi / 2, 0.0, 0.0]

    def section_properties(
        geom_type: str, geom_size: list[float]
    ) -> tuple[float, float, float]:
        # Match plugin/cable.cc section property computations using geom_size.
        if geom_type in ("cylinder", "capsule"):
            r = geom_size[0]
            j = np.pi * r**4 / 2.0
            iy = iz = np.pi * r**4 / 4.0
            return j, iy, iz
        if geom_type == "box":
            h = geom_size[1]
            w = geom_size[2]
            a = max(h, w)
            b = min(h, w)
            j = a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - (b**4) / (a**4) / 12.0))
            iy = (2.0 * w) ** 3 * (2.0 * h) / 12.0
            iz = (2.0 * h) ** 3 * (2.0 * w) / 12.0
            return j, iy, iz
        return 0.0, 0.0, 0.0

    spec = mj.MjSpec()
    root = spec.worldbody.add_body(name=f"{name_prefix}:root", pos=base_pos)
    if add_freejoint:
        root.add_joint(name=f"{name_prefix}:free", type=mj.mjtJoint.mjJNT_FREE)

    first_joint_idx = 1 if n_segments > 1 else None
    last_joint_idx = n_segments - 1 if n_segments > 1 else None

    first_body_idx = 0
    last_body_idx = n_segments - 1

    parent = root
    for i in range(n_segments):
        if i == 0:
            body = parent.add_body(name=f"{name_prefix}:Bfirst")
        else:
            if i == last_body_idx:
                body_name = f"{name_prefix}:Blast"
            else:
                body_name = f"{name_prefix}:B{i}"
            body = parent.add_body(
                name=body_name,
                pos=[0.0, segment_length, 0.0],
            )
            joint_pos = [0.0, -segment_length / 2.0, 0.0]

        if geom_type in ("cylinder", "capsule"):
            geom_size = [radius, segment_length / 2.0, 0]
        elif geom_type == "sphere":
            geom_size = [radius, 0.0, 0.0]
        else:
            geom_size = [radius, segment_length / 2.0, radius]

        # Match plugin stiffness behavior: k = (J*G)/L for twist, (Iy*E)/L & (Iz*E)/L for bend.
        if i > 0:
            j, iy, iz = section_properties(geom_type, geom_size)
            length = max(segment_length, 1e-9)
            k_twist = (j * twist) / length
            k_bend_y = (iy * bend) / length
            k_bend_z = (iz * bend) / length

            # Ball joint uses a single stiffness; approximate from bend/twist contributions.
            k_ball = (k_bend_y + k_bend_z + k_twist) / 3.0
            if i == first_joint_idx:
                joint_name = f"{name_prefix}:Jfirst"
            elif i == last_joint_idx:
                joint_name = f"{name_prefix}:Jlast"
            else:
                joint_name = f"{name_prefix}:J{i}"
            body.add_joint(
                name=joint_name,
                type=mj.mjtJoint.mjJNT_BALL,
                pos=joint_pos,
                damping=damping,
                armature=armature,
                stiffness=k_ball,
            )

        geom = body.add_geom(
            name=(
                f"{name_prefix}:Gfirst"
                if i == first_body_idx
                else f"{name_prefix}:Glast"
                if i == last_body_idx
                else f"{name_prefix}:G{i}"
            ),
            type=geom_type_map[geom_type],
            size=geom_size,
            euler=geom_euler,
            rgba=rgba,
            mass=mass,
            friction=friction,
            condim=condim,
            solref=solref,
        )
        if geom_type in ("cylinder", "capsule"):
            geom.fromto = [
                0.0,
                -segment_length / 2.0,
                0.0,
                0.0,
                segment_length / 2.0,
                0.0,
            ]

        parent = body

    return spec


def empty(floor: bool = False) -> mj.MjSpec:

    # integrator="RK4"
    # <flag multiccd="enable" />
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
        episode_length=300,
        action_repeat=1,
        action_scale_pos=0.01,
        success_threshold_pos=0.02,
        impl="warp",
    )


def _build_model() -> mj.MjModel:
    scene = empty(floor=True)
    scene.option.cone = mj.mjtCone.mjCONE_PYRAMIDAL

    # Target (goal) mocap body.
    b_goal = scene.worldbody.add_body(name="goal", mocap=True, pos=[0.1, 0.1, 0.1])
    b_goal.add_geom(
        name="goal_geom",
        type=mj.mjtGeom.mjGEOM_SPHERE,
        size=[0.015, 0.0, 0.0],
        contype=0,
        conaffinity=0,
        rgba=[1, 0, 0, 1],
    )
    b_goal.add_site(name="goal", size=[0.01], rgba=[1, 0, 0, 1])

    # Mocap handle.
    mocap_pos = [0.0, 0.0, 0.2]
    b_mocap = scene.worldbody.add_body(name="mocap", mocap=True, pos=mocap_pos)
    b_mocap.add_site(name="mocap_site")
    b_mocap.add_geom(
        name="mocap",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
    )

    # Cable welded to the mocap site.
    cable_count = 20
    cable_size = 0.5
    segment_length = cable_size / cable_count
    seg5_offset = segment_length * 5.5
    cable = mjx_cable(
        twist=8_000_000, bend=30_000_000, count=cable_count, size=cable_size
    )

    cable.body("cable:B10").add_site(name="cable_seg10", pos=[0.0, 0.0, 0.0])

    b = cable.body("cable:Bfirst")
    b.add_site(name="keypoint")

    scene.worldbody.add_frame(
        name="cable",
        pos=[mocap_pos[0], mocap_pos[1] - seg5_offset, mocap_pos[2]],
    ).attach_body(cable.worldbody.first_body())

    _f = scene.worldbody.add_frame(name="gripper")

    from robot_descriptions import robotiq_2f85_mj_description

    gripper = mj.MjSpec.from_file(robotiq_2f85_mj_description.MJCF_PATH)

    _b = gripper.worldbody.first_body()
    _b.add_site(name="gripper")
    _b.add_freejoint()

    _f.attach_body(_b, prefix="gripper/")

    # scene.add_equality(
    #     name="cable_weld",
    #     type=mj.mjtEq.mjEQ_WELD,
    #     objtype=mj.mjtObj.mjOBJ_SITE,
    #     name1="gripper/gripper",
    #     name2="mocap_site",
    #     solref=[0.0000000001, 1],
    # )

    # gripper

    # scene.add_key(
    #     name="init",
    #     qpos=_parse_float_list(
    #         "0 -0.0897495 -0.0145742 0.748817 0.662777 0 0 1 -0.0008592 0 0 0.999994 -0.00347 0 0 0.999968 -0.00804724 0 0 0.999885 -0.015183 0 0 0.99966 -0.0260866 0 0 0.999079 -0.0429155 0 0 0.997601 -0.0692315 0 0 0.993878 -0.110486 0 0 0.984766 -0.173883 0 0 0.964201 -0.265174 0 0 0.967627 -0.252385 0 0 0.985699 -0.168517 0 0 0.994056 -0.10887 0 0 0.997627 -0.0688574 0 0 0.999099 -0.0424344 0 0 0.999689 -0.0249255 0 0 0.999912 -0.0132849 0 0 0.999984 -0.00574447 0 0 0.999999 -0.00142357 0 0"
    #     ),
    #     qvel=_parse_float_list(
    #         "0 -0.000304401 -0.000191022 -0.00575935 0 0 6.86306e-05 0 0 0.000274173 0 0 0.000619598 0 0 0.00111146 0 0 0.00174564 0 0 0.0024591 0 0 0.00301178 0 0 0.00274826 0 0 0.00025122 0 0 -0.00658249 0 0 -0.00679669 0 0 -0.000275916 0 0 0.00234904 0 0 0.00279081 0 0 0.00232595 0 0 0.00161242 0 0 0.000939918 0 0 0.00042389 0 0 0.000106391 0 0"
    #     ),
    #     mpos=_parse_float_list("0.1 0.1 0.1 0 0 0.2"),
    #     mquat=_parse_float_list("1 0 0 0 1 0 0 0"),
    # )

    return scene.compile()


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


def _format_final_path(base_path: str) -> str:
    if not base_path:
        return ""
    if "{" in base_path and "}" in base_path:
        try:
            return base_path.format(step="final", eval="final")
        except (KeyError, ValueError):
            pass
    stem, ext = os.path.splitext(base_path)
    if not ext:
        ext = ".gif"
    return f"{stem}_final{ext}"


def _rollout_trajectory(
    env: mjx_env.MjxEnv,
    steps: int,
    seed: int,
    policy=None,
    use_jit: bool = True,
) -> mjx_env.State:
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng)

    def step_fn(carry, _):
        state, key = carry
        key, step_key = jax.random.split(key)
        if policy is None:
            action = jp.zeros((env.action_size,), dtype=jp.float32)
        else:
            action, _ = policy(state.obs, step_key)
        state = env.step(state, action)
        return (state, key), state

    def run_scan(init_state, init_key):
        return jax.lax.scan(step_fn, (init_state, init_key), None, length=steps)

    if use_jit:
        run_scan = jax.jit(run_scan)

    (_final_state, _), traj = run_scan(state, rng)

    # Prepend initial state to match rollout length.
    traj = jax.tree_util.tree_map(
        lambda x0, xs: jp.concatenate([x0[None, ...], xs], axis=0), state, traj
    )
    return traj


def _trajectory_to_list(traj: mjx_env.State) -> list[mjx_env.State]:
    leaves = jax.tree_util.tree_leaves(traj)
    length = int(leaves[0].shape[0]) if leaves else 0
    states: list[mjx_env.State] = []
    for i in range(length):
        states.append(jax.tree_util.tree_map(lambda x: x[i], traj))
    return states


def render_rollout(
    env: mjx_env.MjxEnv,
    steps: int,
    path: str,
    seed: int = 0,
    fps: int = 30,
    policy=None,
    use_jit: bool = True,
    frame_skip: int = 1,
    height: int = 480,
    width: int = 640,
) -> None:
    frame_skip = max(int(frame_skip), 1)
    traj = _rollout_trajectory(env, steps, seed, policy=policy, use_jit=use_jit)
    if frame_skip > 1:
        traj = jax.tree_util.tree_map(lambda x: x[::frame_skip], traj)
    traj = jax.device_get(traj)
    trajectory = _trajectory_to_list(traj)
    frames = env.render(trajectory, height=height, width=width)
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
        # Disable CCD to avoid overflow warnings in MuJoCo.
        # self._mj_model.opt.disableflags |= int(mj.mjtDisableBit.mjDSBL_NATIVECCD)
        # self._mj_model.opt.enableflags &= ~int(mj.mjtEnableBit.mjENBL_MULTICCD)
        self._mj_model.opt.timestep = float(config.sim_dt)
        self._mjx_model = mjx.put_model(self._mj_model, impl=config.impl)
        self._episode_length = config.episode_length
        self._action_scale_pos = config.action_scale_pos
        self._success_threshold_pos = config.success_threshold_pos

        self._goal_site_id = _as_index(self._mj_model.site("goal").id)
        self._cable_tip_site_id = _as_index(self._mj_model.site("keypoint").id)
        self._ee_mocap_id = _as_index(self._mj_model.body("mocap").mocapid)
        self._goal_mocap_id = _as_index(self._mj_model.body("goal").mocapid)
        self._mocap_body_id = _as_index(self._mj_model.body("mocap").id)
        self._cable_root_jnt_id = _as_index(self._mj_model.joint("cable:free").id)
        self._cable_root_qpos_adr = _as_index(
            self._mj_model.jnt_qposadr[self._cable_root_jnt_id]
        )

        self._workspace_low = jp.array([-0.3, -0.3, -0.1])
        self._workspace_high = jp.array([0.3, 0.3, 0.3])
        key_id = 0
        self._key_qpos = jp.array(self._mj_model.key_qpos[key_id])
        self._key_qvel = jp.array(self._mj_model.key_qvel[key_id])
        self._key_mpos = jp.array(self._mj_model.key_mpos[key_id]).reshape(
            self._mj_model.nmocap, 3
        )
        self._key_mquat = jp.array(self._mj_model.key_mquat[key_id]).reshape(
            self._mj_model.nmocap, 4
        )

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_goal, rng_delta = jax.random.split(rng, 3)
        goal = jax.random.uniform(
            rng_goal,
            shape=(3,),
            minval=self._workspace_low,
            maxval=self._workspace_high,
        )

        data = mjx.make_data(
            self._mj_model,
            impl=self._mjx_model.impl.value,
            naconmax=256,
            njmax=256,
            # self._mj_model, impl=self._mjx_model.impl.value, nconmax=10 * 256
        )

        mocap_base = self._key_mpos[self._ee_mocap_id]
        mocap_pos = jax.random.uniform(
            rng_delta,
            shape=(3,),
            minval=self._workspace_low,
            maxval=self._workspace_high,
        )
        delta = mocap_pos - mocap_base

        qpos = self._key_qpos
        qpos = qpos.at[self._cable_root_qpos_adr : self._cable_root_qpos_adr + 3].add(
            delta
        )
        qvel = self._key_qvel

        mpos = self._key_mpos
        mpos = mpos.at[self._ee_mocap_id].set(mocap_pos)
        mpos = mpos.at[self._goal_mocap_id].set(goal)
        mquat = self._key_mquat
        mquat = mquat.at[self._goal_mocap_id].set(jp.array([1.0, 0.0, 0.0, 0.0]))

        data = data.replace(qpos=qpos, qvel=qvel, mocap_pos=mpos, mocap_quat=mquat)

        data = mjx.forward(self._mjx_model, data)

        obs = self._get_obs(data)
        info = {"rng": rng, "step": jp.array(0)}
        reward = jp.array(0.0)
        done = jp.array(0.0)
        metrics = {
            "reward": reward,
            "pos_error": jp.linalg.norm(obs),
        }
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        dpos = jp.clip(action[:3], -1.0, 1.0) * self._action_scale_pos
        target = state.data.mocap_pos[self._ee_mocap_id] + dpos
        target = jp.clip(target, self._workspace_low, self._workspace_high)

        data = state.data.replace(
            mocap_pos=state.data.mocap_pos.at[self._ee_mocap_id].set(target),
        )

        data = mjx_env.step(self._mjx_model, data, data.ctrl, self.n_substeps)

        obs = self._get_obs(data)
        pos_err = jp.linalg.norm(obs)
        reward = -pos_err
        done = pos_err < self._success_threshold_pos
        timeout = state.info["step"] >= self._episode_length
        done = done | timeout

        info = {**state.info, "step": state.info["step"] + 1}
        metrics = {
            **state.metrics,
            "reward": reward,
            "pos_error": pos_err,
        }
        return mjx_env.State(data, obs, reward, done.astype(float), metrics, info)

    def _get_obs(self, data: mjx.Data) -> jax.Array:
        tip_pos = data.site_xpos[self._cable_tip_site_id]
        goal_pos = data.site_xpos[self._goal_site_id]
        return goal_pos - tip_pos

    @property
    def action_size(self) -> int:
        return 3

    @property
    def observation_size(self) -> int:
        return 3

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


def view(model: mj.MjModel):
    import glfw
    import mujoco.viewer

    m = model
    d = mj.MjData(m)

    close = False

    def cb(key: int) -> None:
        if key is glfw.KEY_SPACE:
            global close
            close = True
        if key is glfw.KEY_PERIOD:
            d.ctrl = np.array([255])

    with mujoco.viewer.launch_passive(model=m, data=d, key_callback=cb) as viewer:
        # for i, jn in enumerate(jnts):
        #     d.joint(jn).qpos = ref[i]
        # for i, jn in enumerate(acts):
        #     d.actuator(jn).ctrl = ref[i]
        while not close:
            step_start = time.time()

            # step simulation one time step
            mj.mj_step(m, d)

            viewer.sync()

            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


@dataclass
class TrainArgs:
    num_timesteps: int = 200_000
    num_evals: int = 10
    episode_length: int = 150
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
    batch_size: int = 4096
    seed: int = 0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    deterministic_eval: bool = False
    metrics_plot_path: str = "testing/envs/mocap_1.png"
    render_path: str = "testing/envs/mocap_1_rollout.gif"
    render_steps: int = 150
    render_fps: int = 30
    render_impl: str = "jax"
    render_use_jit: tyro.conf.FlagConversionOff[bool] = True
    render_frame_skip: int = 1
    render_height: int = 480
    render_width: int = 640
    render_eval_rollouts: tyro.conf.FlagConversionOff[bool] = True
    render_eval_path: str = ""
    render_after_training: tyro.conf.FlagConversionOff[bool] = True
    render_final_path: str = ""


def main() -> None:

    VIEW = True

    args = tyro.cli(TrainArgs)

    print("config...")

    cfg = default_config()
    cfg.episode_length = args.episode_length
    cfg.action_repeat = args.action_repeat

    print("env...")
    env = MocapReach(cfg)
    render_env = env
    if args.render_impl != cfg.impl:
        render_cfg = default_config()
        render_cfg.ctrl_dt = cfg.ctrl_dt
        render_cfg.sim_dt = cfg.sim_dt
        render_cfg.episode_length = cfg.episode_length
        render_cfg.action_repeat = cfg.action_repeat
        render_cfg.action_scale_pos = cfg.action_scale_pos
        render_cfg.success_threshold_pos = cfg.success_threshold_pos
        render_cfg.impl = args.render_impl
        render_env = MocapReach(render_cfg)

    if VIEW:
        view(env.mj_model)
        quit()

    plot_keys = [
        "eval/episode_reward",
        "training/total_loss",
        "training/sps",
    ]
    metrics_history: dict[str, list[float]] = {key: [] for key in plot_keys}
    metrics_history["step"] = []
    eval_rollout_index = 0

    def progress_fn(step: int, metrics: dict) -> None:
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
            render_env,
            steps=args.render_steps,
            path=eval_path,
            seed=seed,
            fps=args.render_fps,
            policy=policy,
            use_jit=args.render_use_jit,
            frame_skip=args.render_frame_skip,
            height=args.render_height,
            width=args.render_width,
        )
        print(f"saved eval rollout to {eval_path}")

    print("train def...")
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

    print("train run...")
    make_inference_fn, params, metrics = train_fn(
        environment=env,
        wrap_env_fn=wrap_for_mjx_training,
    )

    print("training metrics:", metrics)

    if args.render_after_training:
        policy = make_inference_fn(params, deterministic=args.deterministic_eval)
        final_path = args.render_final_path or _format_final_path(args.render_path)
        if final_path:
            render_rollout(
                render_env,
                steps=args.render_steps,
                path=final_path,
                seed=args.seed + 10_000,
                fps=args.render_fps,
                policy=policy,
                use_jit=args.render_use_jit,
                frame_skip=args.render_frame_skip,
                height=args.render_height,
                width=args.render_width,
            )
            print(f"saved final rollout to {final_path}")

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
    final_pos_err = np.linalg.norm(final_obs)
    print(f"final errors (unbatched): pos={float(final_pos_err):.4f}")


if __name__ == "__main__":
    main()
