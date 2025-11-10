import jax
import jax.numpy as jp
from brax import base
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
from typing import Any


class XPose(PipelineEnv):
    """Simple 3D position control environment in Brax with fixed target."""

    def __init__(self, backend='generalized', **kwargs):
        # Create the MJCF XML string
        xml_string = """
        <mujoco>
            <option timestep="0.01" iterations="1" ls_iterations="4">
                <flag contact="disable" eulerdamp="disable"/>
            </option>

            <visual>
                <rgba haze="0.15 0.25 0.35 1"/>
                <quality shadowsize="4096"/>
                <map stiffness="700" shadowscale="0.5" fogstart="1" fogend="150" zfar="40" haze="1"/>
            </visual>

            <asset>
                <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="1 1 1" width="512" height="512"/>
                <texture name="texplane" type="2d" builtin="checker" rgb1=".2 .3 .4" rgb2=".1 0.15 0.2"
                    width="512" height="512" mark="cross" markrgb=".8 .8 .8"/>
                <material name="matplane" reflectance="0.3" texture="texplane" texrepeat="10 10" texuniform="true"/>
            </asset>

            <worldbody>
                <camera name="lookat" mode="targetbody" target="agent" pos="0.2 0.2 0.2"/>
                <body name="agent" gravcomp="1">
                    <joint name="x" type="slide" axis="1 0 0" range="-1 1"/>
                    <joint name="y" type="slide" axis="0 1 0" range="-1 1"/>
                    <joint name="z" type="slide" axis="0 0 1" range="-1 1"/>
                    <geom name="agent" type="box" size="0.01 0.01 0.01" contype="0" conaffinity="0" rgba="0.5 0 0 0.3"/>
                </body>
            </worldbody>
            
            <actuator>
                <position name="x" joint="x" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="y" joint="y" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
                <position name="z" joint="z" ctrlrange="-1 1" kp="10" kv="100" ctrllimited="true"/>
            </actuator>
        </mujoco>
        """
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_string)
            xml_path = f.name

        sys = mjcf.load(xml_path)
        os.unlink(xml_path)

        n_frames = 2
        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 4

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        # Print bodies to confirm successful load
        for i in range(self.sys.mj_model.nbody):
            print("\t", self.sys.mj_model.body(i).name)

        # Fixed target position in world coordinates
        self._target = jp.array([0.0, 0.0, 0.0])

        # Target position bounds (for termination)
        self._target_bounds = jp.array([[-0.8, -0.8, 0.1], [0.8, 0.8, 0.8]])
        self._timeout = 500  # timesteps

    # ---------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> State:
        """Resets the environment to an initial state."""
        rng1, rng2 = jax.random.split(rng)

        # Initialize q with small random noise around zero
        q = jax.random.uniform(rng1, (self.sys.q_size(),), minval=-0.01, maxval=0.01)

        # Randomize agent starting position
        q = q.at[0].set(jax.random.uniform(rng1, minval=-0.5, maxval=0.5))  # x
        q = q.at[1].set(jax.random.uniform(rng1, minval=-0.5, maxval=0.5))  # y
        q = q.at[2].set(jax.random.uniform(rng1, minval=0.1, maxval=0.5))   # z

        # Small random velocity
        qd = jax.random.uniform(rng2, (self.sys.qd_size(),), minval=-0.01, maxval=0.01)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done = jp.zeros(2)
        metrics = {
            # 'target_pos': self._target,
            # 'agent_pos': q[:3],
            # 'step_count': jp.array(0),
            # 'distance': jp.linalg.norm(q[:3] - self._target),
        }

        return State(pipeline_state, obs, reward, done, metrics)

    # ---------------------------------------------------------------------

    def step(self, state: State, action: jax.Array) -> State:
        """Run one timestep of the environment's dynamics."""
        action_min = self.sys.actuator.ctrl_range[:, 0]
        action_max = self.sys.actuator.ctrl_range[:, 1]
        scaled_action = (action + 1) * (action_max - action_min) * 0.5 + action_min

        pipeline_state = self.pipeline_step(state.pipeline_state, scaled_action)

        agent_pos = pipeline_state.q[:3]
        target_pos = self._target

        # Reward: negative distance, with small bonus for being close
        dist_to_target = jp.linalg.norm(agent_pos - target_pos)
        reward = -dist_to_target + 1.0 * (dist_to_target < 0.05)

        obs = self._get_obs(pipeline_state)

        # current_step = state.metrics['step_count'] + 1
        # timeout = current_step >= self._timeout
        out_of_bounds = (
            jp.any(agent_pos < self._target_bounds[0] + 0.1)
            | jp.any(agent_pos > self._target_bounds[1] + 0.1)
        )
        nan_condition = jp.isnan(pipeline_state.q).any() | jp.isnan(pipeline_state.qd).any()

        done = (nan_condition | out_of_bounds).astype(float)

        metrics = {
            # 'target_pos': target_pos,
            # 'agent_pos': agent_pos,
            # 'step_count': current_step,
            # 'distance': dist_to_target,
            # 'timeout': timeout,
            # 'out_of_bounds': out_of_bounds,
        }

        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics
        )

    # ---------------------------------------------------------------------

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe agent position, target position, and relative position."""
        agent_pos = pipeline_state.q[:3]
        relative_pos = agent_pos - self._target
        return jp.concatenate([agent_pos, self._target, relative_pos])

    # ---------------------------------------------------------------------

    @property
    def action_size(self):
        return 3

    @property
    def observation_size(self):
        return 9  # agent_pos(3) + target_pos(3) + relative_pos(3)
