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
        episode_length=150,
        action_repeat=1,
        action_scale_pos=0.02,
        success_threshold_pos=0.02,
        impl="warp",
    )


def _build_model() -> mj.MjModel:
    scene = empty(floor=False)
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
    b_mocap = scene.worldbody.add_body(name="mocap", mocap=True, pos=[0.0, 0.0, 0.2])
    b_mocap.add_site(name="mocap_site", size=[0.008], rgba=[0, 1, 0, 0.4])

    # Free body to be moved.
    b_obj = scene.worldbody.add_body(name="object", pos=[0.0, 0.0, 0.2])
    b_obj.add_freejoint()
    b_obj.add_geom(
        name="object_geom",
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        rgba=[0.2, 0.6, 0.9, 1.0],
    )
    b_obj.add_site(name="object_site", size=[0.01], rgba=[0.2, 0.6, 0.9, 1.0])

    # Weld object to the mocap handle via sites.
    scene.add_equality(
        name="object_weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_SITE,
        name1="object_site",
        name2="mocap_site",
        solref=[0.000001, 2],
    )

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

        self._object_site_id = _as_index(self._mj_model.site("object_site").id)
        self._goal_site_id = _as_index(self._mj_model.site("goal").id)
        self._ee_mocap_id = _as_index(self._mj_model.body("mocap").mocapid)
        self._goal_mocap_id = _as_index(self._mj_model.body("goal").mocapid)
        self._mocap_body_id = _as_index(self._mj_model.body("mocap").id)

        self._workspace_low = jp.array([-0.3, -0.3, -0.1])
        self._workspace_high = jp.array([0.3, 0.3, 0.3])
        self._mocap_home_pos = jp.array(self._mj_model.body_pos[self._mocap_body_id])

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, rng_goal = jax.random.split(rng, 2)
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

        mocap_pos = jp.clip(
            self._mocap_home_pos, self._workspace_low, self._workspace_high
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._ee_mocap_id].set(mocap_pos),
            mocap_quat=data.mocap_quat.at[self._ee_mocap_id].set(
                jp.array([1.0, 0.0, 0.0, 0.0])
            ),
        )
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._goal_mocap_id].set(goal),
            mocap_quat=data.mocap_quat.at[self._goal_mocap_id].set(
                jp.array([1.0, 0.0, 0.0, 0.0])
            ),
        )

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
        obj_pos = data.site_xpos[self._object_site_id]
        goal_pos = data.site_xpos[self._goal_site_id]
        return goal_pos - obj_pos

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


@dataclass
class TrainArgs:
    num_timesteps: int = 2_00_000
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
    metrics_plot_path: str = "testing/envs/mocap_0.png"
    render_path: str = "testing/envs/mocap_0_rollout.gif"
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
