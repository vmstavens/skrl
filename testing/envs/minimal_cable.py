import time
from typing import Optional

import imageio.v2 as imageio
import jax
import jax.numpy as jp
import mujoco as mj
import mujoco.viewer
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env

_BENT_QPOS = (
    "0 0.998504 -6.4184e-19 0.0546818 4.2918e-18 0.998842 1.04001e-18 "
    "0.0481006 4.65137e-18 0.999118 2.6788e-18 0.0419877 5.5544e-18 "
    "0.999339 3.47673e-18 0.0363647 6.55935e-18 0.999512 2.3655e-18 "
    "0.0312313 5.94352e-18 0.999647 3.43243e-18 0.0265736 7.08841e-18 "
    "0.99975 4.64495e-18 0.0223702 7.48067e-18 0.999827 2.57826e-18 "
    "0.0185962 6.27845e-18 0.999884 2.03928e-18 0.0152264 4.8484e-18 "
    "0.999925 2.13929e-18 0.0122369 3.85855e-18 0.999954 1.75913e-18 "
    "0.0096067 3.0255e-18 0.999973 1.83893e-18 0.00731777 2.54826e-18 "
    "0.999986 8.94681e-19 0.00535557 1.31129e-18 0.999993 3.41442e-19 "
    "0.00370878 3.93121e-19 0.999997 2.01012e-19 0.00236919 4.72375e-19 "
    "0.999999 -1.74867e-20 0.00133118 9.57574e-20 1 -9.97713e-20 "
    "0.000591306 -1.47549e-19 1 -1.98974e-20 0.000147801 2.52819e-21"
)
_BENT_QVEL = (
    "0 1.9393e-16 -9.0442e-06 3.22404e-17 1.11784e-16 -8.34318e-06 "
    "3.33174e-17 -9.45701e-17 -7.62616e-06 4.73589e-18 -2.69248e-16 "
    "-6.90682e-06 -3.46012e-17 -3.51806e-16 -6.19534e-06 -6.43823e-17 "
    "-3.3775e-16 -5.49921e-06 -6.80477e-17 -2.70417e-16 -4.82405e-06 "
    "-3.50925e-17 -2.58612e-16 -4.17431e-06 -5.4011e-17 -1.83978e-16 "
    "-3.55387e-06 -3.31808e-17 -1.15477e-16 -2.96648e-06 -6.67119e-18 "
    "-7.60188e-17 -2.41613e-06 1.50717e-17 -2.82536e-17 -1.90726e-06 "
    "4.40649e-17 -2.01703e-17 -1.44484e-06 3.0712e-17 -1.07938e-17 "
    "-1.03452e-06 1.96359e-17 -6.67439e-18 -6.82523e-07 8.32128e-18 "
    "-3.23017e-18 -3.95647e-07 1.67609e-18 -2.15466e-18 -1.8115e-07 "
    "1.48849e-18 -1.39471e-18 -4.66503e-08 -3.55623e-19"
)
_BENT_CTRL = "0"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.002,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        vision=False,
    )


def _parse_float_list(values: str) -> list[float]:
    return [float(item) for item in values.split()]


def mjs_cable(
    model_name: str = "cable",
    prefix: str = "cable:",
    curve: str = "0 s 0",
    count: str = "20 1 1",
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    size: str = 1,
    segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: str = "0.8 0.2 0.2 1",
    initial: str = "none",
    # initial: str = "free",
) -> mj.MjSpec:
    xml = f"""
<mujoco model="{model_name}">
    <extension>
        <plugin plugin="mujoco.elasticity.cable"/>
    </extension>
    <worldbody>
        <body name="{prefix}root" pos="0 0 0">
            <composite prefix="{prefix}" type="cable" curve="{curve}" count="{count}" size="{size}" initial="{initial}">
                <plugin plugin="mujoco.elasticity.cable">
                    <config key="twist" value="{twist}"/>
                    <config key="bend" value="{bend}"/>
                    <config key="vmax" value="{vmax}"/>
                </plugin>
                <joint kind="main" damping="1e-2" armature="0.001"/>
                <geom type="capsule" size="{segment_size}" rgba="{rgba}" mass="{mass}"/>
            </composite>
        </body>
    </worldbody>
</mujoco>
    """
    return mj.MjSpec.from_string(xml)


def _build_model() -> mj.MjModel:
    xml = """
<mujoco model="minimal_cable">
    <extension>
        <plugin plugin="mujoco.elasticity.cable"/>
    </extension>
    <option timestep="0.002" gravity="0 0 -9.82" integrator="implicitfast" solver="Newton"/>
    <worldbody>
        <camera name="cam" pos="1.278 0.185 -0.059" xyaxes="0.089 0.996 0.000 -0.118 0.011 0.993"/>
        <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
        <body name="slider" pos="0 0.2 0.1">
            <joint name="x" type="slide" axis="1 0 0" range="-0.3 0.3"/>
            <geom type="box" size="0.02 0.02 0.02" rgba="0.7 0.7 0.7 0"/>
        </body>
    </worldbody>
</mujoco>
    """
    scene = mj.MjSpec.from_string(xml)

    scene.add_actuator(
        name="x",
        target="x",
        trntype=mj.mjtTrn.mjTRN_JOINT,
        ctrlrange=[-0.3, 0.3],
    ).set_to_position(kp=100, kv=20)

    # cable = mjs_cable(twist=10e12, bend=10e12)
    # cable = mjs_cable(twist=1_000_000 * 1000, bend=1_000_000 * 1000)
    cable = mjs_cable(twist=1_000_000 * 1000, bend=1_000_000 * 1000)
    scene.worldbody.add_frame(pos=[0, 0, 0]).attach_body(cable.worldbody.first_body())

    scene.add_key(
        name="bent",
        time=19.742,
        qpos=_parse_float_list(_BENT_QPOS),
        qvel=_parse_float_list(_BENT_QVEL),
        ctrl=_parse_float_list(_BENT_CTRL),
    )

    return scene.compile()


class MinimalCable(mjx_env.MjxEnv):
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict] = None,
    ) -> None:
        super().__init__(config, config_overrides=config_overrides)

        self.episode_length = self._config.episode_length

        self._mj_model = _build_model()
        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = "minimal_cable.xml"
        self._key_id: Optional[int] = None
        try:
            self._key_id = self._mj_model.key("bent").id
        except Exception:
            self._key_id = None

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = jp.zeros(self.mj_model.nq)
        qvel = jp.zeros(self.mj_model.nv)
        ctrl = jp.zeros(self.mj_model.nu)
        act = jp.zeros(self.mj_model.na)
        if self._key_id is not None:
            qpos = jp.array(self._mj_model.key_qpos[self._key_id])
            qvel = jp.array(self._mj_model.key_qvel[self._key_id])
            ctrl = jp.array(self._mj_model.key_ctrl[self._key_id])
            if self.mj_model.na:
                act = jp.array(self._mj_model.key_act[self._key_id])
        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl, act=act)
        obs = data.qpos
        reward = jp.zeros(2)
        done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, {}, {})

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        obs = data.qpos
        reward = jp.zeros(2)
        done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, state.metrics, state.info)

    @property
    def observation_size(self) -> int:
        return self.mj_model.nq

    @property
    def action_size(self) -> int:
        return self.mj_model.nu

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mj.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model


if __name__ == "__main__":
    MODE = "gui_rollout"  # options: "rollout", "gui_rollout", "gui_render"

    env = MinimalCable()
    if MODE in ("gui_rollout", "gui_render"):
        model = env.mj_model
        data = mj.MjData(model)
        if env._key_id is not None:
            mj.mj_resetDataKeyframe(model, data, env._key_id)
            mj.mj_forward(model, data)
        with mujoco.viewer.launch_passive(model=model, data=data) as viewer:
            renderer = mujoco.Renderer(model, height=480, width=640)
            frames = []
            if MODE == "gui_render":
                while viewer.is_running():
                    mj.mj_step(model, data)
                    viewer.sync()
            else:
                for _ in range(env.episode_length):
                    if not viewer.is_running():
                        break
                    step_start = time.time()

                    mj.mj_step(model, data)
                    if MODE == "gui_rollout":
                        renderer.update_scene(data, camera="cam")
                        frames.append(renderer.render())
                    viewer.sync()

                    time_until_next_step = model.opt.timestep - (
                        time.time() - step_start
                    )
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)

                if MODE == "gui_rollout":
                    try:
                        output_path = "minimal_cable_gui.mp4"
                        imageio.mimsave(output_path, frames, fps=1.0 / env.dt)
                        print(f"Saved GUI video to {output_path}")
                    except Exception:
                        print(f"Rendered {len(frames)} GUI frames")
    else:
        jit_reset = jax.jit(env.reset)
        jit_step = jax.jit(env.step)

        state = jit_reset(jax.random.PRNGKey(0))
        rollout = [state]

        for _ in range(env.episode_length):
            action = jp.zeros(env.action_size)
            state = jit_step(state, action)
            rollout.append(state)

        frames = env.render(rollout, height=480, width=640, camera="cam")
        try:
            output_path = "minimal_cable_rollout_12.mp4"
            imageio.mimsave(output_path, frames, fps=1.0 / env.dt)
            print(f"Saved rollout video to {output_path}")
        except Exception:
            print(f"Rendered {len(frames)} frames")
