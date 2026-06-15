import collections
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import brax
import brax.envs
import imageio
import jax
import mujoco as mj
import mujoco.viewer
import numpy
import numpy as np
import numpy.lib.npyio
import torch
import torch.nn as nn
from brax import envs
from brax.envs import _envs as _ENV_REGISTRY
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper
from brax.io import model
from brax.training.acme.running_statistics import RunningStatisticsState
from brax.training.networks import FeedForwardNetwork
from jax import numpy as jp
from mujoco_playground import wrapper, wrapper_torch

# from learning.algos.pytorch.ppo import PPO
from .cfg import Cfg

StepData = collections.namedtuple(
    "StepData", ("observation", "logits", "action", "reward", "done", "truncation")
)


def create_data_directory(environment_name: str, session_name: str) -> Path:
    """
    Create a directory structure for storing data related to a specific environment and session.

    Args:
        environment_name (str): The name of the environment.
        session_name (str): The name of the session.

    Returns:
        Path: The path to the created session directory.
    """
    root_dir = Path(__file__).parent.parent
    data_dir = root_dir / "learning" / "data"
    session_dir = data_dir / environment_name / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = session_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"Created directories for session: {session_dir}")
    return session_dir


def get_dimensions_from_params(
    params: Tuple[RunningStatisticsState, Dict],
) -> Tuple[int, int]:
    """
    Extract observation and action dimensions from model parameters.

    Args:
        params (Tuple[RunningStatisticsState, dict]):
            A tuple containing the running statistics state and a dictionary of model parameters.

    Returns:
        Tuple[int, int]: A tuple where the first element is the observation dimension (o_dim)
                         and the second element is the action dimension (a_dim).
    """
    parameter_dict = params[1]["params"]
    hidden_keys: List[str] = list(parameter_dict.keys())
    hidden_first, hidden_last = hidden_keys[0], hidden_keys[-1]

    # Observation dimension
    o_dim = np.array(parameter_dict[hidden_first]["kernel"]).shape[0]

    # Action dimension (last layer output divided by 2)
    a_dim = np.array(parameter_dict[hidden_last]["kernel"]).shape[1] // 2

    return o_dim, a_dim


def get_mlp_architecture_from_params(
    params: Union[Tuple[RunningStatisticsState, Dict], str, Path],
) -> List[int]:
    """
    Extract the MLP architecture dimensions from model parameters.

    Args:
        params (Union[Tuple[RunningStatisticsState, dict], str, Path]):
            Either a tuple containing the running statistics state and a dictionary of model parameters,
            or a string/Path object pointing to the parameter file to be loaded.

    Returns:
        List[int]: A list representing the architecture dimensions where:
                  - Each element corresponds to the output dimension of a hidden layer
                  - The last element is the action dimension (output dimension divided by 2)
    """
    if isinstance(params, str):
        params = Path(params)
    if isinstance(params, Path):
        params = model.load_params(params)

    parameter_dict: dict = params[1]["params"]
    hidden_keys: List[str] = list(parameter_dict.keys())
    dims = []
    for i, hk in enumerate(hidden_keys):
        layer_shape = np.array(parameter_dict[hk]["kernel"]).shape
        if i == len(hidden_keys) - 1:
            dims.append(layer_shape[1])
        else:
            dims.append(layer_shape[0])
    dims[-1] //= 2
    return dims


def generate_demo_video(
    env_name: str,
    policy: FeedForwardNetwork,
    n_steps: int = 500,
    render_every: int = 2,
    camera_name: str = "side",
    backend: str = "mjx",
) -> list[np.ndarray]:
    """
    Generates a demonstration video of a reinforcement learning agent's performance.

    This function creates a video by simulating an environment and rendering frames
    at specified intervals. The agent's actions are determined using a provided
    inference function and trained model parameters.

    Args:
        env_name (str): Name of the Brax environment to simulate.
        policy (FeedForwardNetwork): Policy network for the agent.
        n_steps (int): Number of simulation steps to run. Defaults to 500.
        render_every (int): Frequency of rendering frames (every `render_every` steps).
                            Defaults to 2.
        camera_name (str): Name of the camera to use for rendering. Defaults to "side".

    Returns:
        list[np.ndarray]: A list of rendered video frames, where each frame is a NumPy
                          array representing the image in RGB format.

    Raises:
        ValueError: If the environment is not registered, no cameras are available,
                   or the specified camera is not found.
    """
    # Check if environment is registered
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Environment '{env_name}' is not registered. Available environments: {list(_ENV_REGISTRY.keys())}"
        )

    # Initialize environment
    eval_env = envs.get_environment(env_name, backend=backend)
    if eval_env is None:
        raise ValueError(
            f"Environment '{env_name}' could not be initialized. Check the environment name."
        )

    # Check if environment uses MuJoCo backend
    if not hasattr(eval_env.sys, "mj_model"):
        raise ValueError(
            f"Environment '{env_name}' does not use MuJoCo backend. Video generation only works with MuJoCo environments."
        )

    # Get MuJoCo model and data
    mj_model: mj.MjModel = eval_env.sys.mj_model
    mj_data = mj.MjData(mj_model)

    # Initialize renderer and check cameras
    renderer = mj.Renderer(mj_model)

    # Get available cameras
    available_cameras = []
    for i in range(mj_model.ncam):
        cam_name = (
            mj_model.names[mj_model.name_camadr[i] :].decode("utf-8").split("\x00")[0]
        )
        if cam_name:
            available_cameras.append(cam_name)

    # Check if any cameras exist
    if not available_cameras:
        raise ValueError("No cameras found in the MuJoCo model. Cannot render frames.")

    # Check if requested camera exists
    if camera_name not in available_cameras:
        raise ValueError(
            f"Camera '{camera_name}' not found. Available cameras: {available_cameras}"
        )

    # JIT compile functions for performance
    jit_policy = jax.jit(policy)
    jit_step = jax.jit(eval_env.step)
    jit_reset = jax.jit(eval_env.reset)
    rng = jax.random.PRNGKey(0)

    # Initialize control input
    ctrl = jp.zeros(mj_model.nu)
    rollout = []
    state = jit_reset(rng)
    # Initialize the state
    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_policy(state.obs, act_rng)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)

    frames = eval_env.render(rollout[::render_every], camera=cam_name)

    # Transpose frames to (time, channels, height, width)
    frames = np.transpose(np.array(frames), axes=(0, 3, 1, 2))
    return frames


def save_frames_as_video(
    frames: np.ndarray,
    output_path: Union[str, Path],
    fps: int = 30,
    quality: int = 8,
    format: str = "mp4",
) -> None:
    """
    Save frames as a video file.

    Args:
        frames: numpy array of shape (n_frames, channels, height, width)
        output_path: path to save the video
        fps: frames per second
        quality: video quality (1-10, higher is better)
        format: video format ('mp4', 'gif', 'avi', etc.)
    """
    # Convert frames from (n_frames, channels, height, width) to (n_frames, height, width, channels)
    video_frames = np.transpose(frames, (0, 2, 3, 1))

    # Convert to uint8 if needed
    if video_frames.dtype != np.uint8:
        if video_frames.max() <= 1.0:
            video_frames = (video_frames * 255).astype(np.uint8)
        else:
            video_frames = video_frames.astype(np.uint8)

    # Create parent directories if they don't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the video
    writer = imageio.get_writer(output_path, fps=fps, quality=quality, format=format)
    for frame in video_frames:
        writer.append_data(frame)
    writer.close()


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
    """Map a function over each field in StepData."""
    items = {}
    keys = sds[0]._asdict().keys()
    for k in keys:
        items[k] = f(*[sd._asdict()[k] for sd in sds])
    return StepData(**items)


def eval_unroll(agent: nn.Module, env: brax.envs.Env, length: int):
    """Return number of episodes and average reward for a single unroll."""
    observation = env.reset()
    episodes = torch.zeros((), device=agent.device)
    episode_reward = torch.zeros((), device=agent.device)
    for _ in range(length):
        _, action = agent.get_logits_action(observation)
        observation, reward, done, _ = env.step(agent(action))
        # observation, reward, done, _ = env.step(PPO.dist_postprocess(action))
        episodes += torch.sum(done)
        episode_reward += torch.sum(reward)
    return episodes, episode_reward / episodes


class Agent(ABC, nn.Module):
    """Abstract base class for all RL agents."""

    @abstractmethod
    def get_action(self, observation: torch.Tensor) -> dict:
        """Get action from observation."""
        pass

    @abstractmethod
    def update(self, batch_data):
        """Perform one training update."""
        pass

    @abstractmethod
    def update_normalization(self, observations):
        """Update observation normalization stats if needed."""
        pass


def train_unroll(
    agent: Agent,
    env: brax.envs.Env,
    observation: torch.Tensor,
    num_unrolls: int,
    unroll_length: int,
):
    """Return step data over multple unrolls."""
    sd = StepData([], [], [], [], [], [])
    for _ in range(num_unrolls):
        one_unroll = StepData([observation], [], [], [], [], [])
        for _ in range(unroll_length):
            print(f"{observation.shape=}")
            action = agent.get_action(observation)
            # logits, action = agent.get_action(observation)
            # logits, action = agent.get_logits_action(observation)
            # logits, action = agent.get_logits_action(observation)
            observation, reward, done, info = env.step(agent(action))
            # observation, reward, done, info = env.step(PPO.dist_postprocess(action))
            one_unroll.observation.append(observation)
            # one_unroll.logits.append(logits)
            one_unroll.action.append(action)
            one_unroll.reward.append(reward)
            one_unroll.done.append(done)
            one_unroll.truncation.append(info["truncation"])
        one_unroll = sd_map(torch.stack, one_unroll)
        sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
    td = sd_map(torch.stack, sd)
    return observation, td


def evaluate(agent: Agent, env, eval_length: int):
    """Evaluation helper function."""
    observation = env.reset()
    episodes = torch.zeros((), device=agent.device)
    total_reward = torch.zeros((), device=agent.device)

    for _ in range(eval_length):
        action = agent.get_action(observation)
        observation, reward, done, _ = env.step(action)
        episodes += torch.sum(done)
        total_reward += torch.sum(reward)

    return episodes, total_reward / episodes if episodes > 0 else 0


def _jax_to_torch(tensor):
    import torch.utils.dlpack as tpack  # pytype: disable=import-error # pylint: disable=import-outside-toplevel

    tensor = tpack.from_dlpack(tensor)
    return tensor


def _torch_to_jax(tensor):
    from jax.dlpack import from_dlpack  # pylint: disable=import-outside-toplevel

    tensor = from_dlpack(tensor)
    return tensor


def train(
    agent: Agent,
    env: brax.envs.PipelineEnv,
    # env: brax.envs.Env,
    cfg: Cfg,
    progress_cb: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> Agent:
    """
    Generic training loop for reinforcement learning agents.

    Args:
        agent: An instance of an Agent subclass (PPO, TD3, etc.)
        env_name: Name of the Brax environment
        num_envs: Number of parallel environments
        episode_length: Maximum length of an episode
        num_timesteps: Total number of timesteps to train for
        eval_frequency: Evaluate every N updates
        unroll_length: Length of each unroll (trajectory segment)
        batch_size: Total batch size across all minibatches
        num_minibatches: Number of minibatches per update
        num_update_epochs: Number of passes over collected data
        progress_fn: Callback for reporting progress
        backend: Brax physics backend ("spring", "mjx", etc.)
        device: Device to use for training ("cuda" or "cpu")

    Returns:
        The trained agent
    """

    # Create environment
    # env = envs.create(env_name, batch_size=num_envs, episode_length=episode_length)

    # from brax.envs import wrappers

    # env = wrapper.brax_training.EpisodeWrapper(
    #     env, cfg.episode_length, action_repeat=cfg.action_repeat
    # )

    # Get environment specs
    # env = envs.create(
    #     cfg.env_name,
    #     batch_size=cfg.num_envs,
    #     episode_length=cfg.episode_length,
    #     backend="spring",
    # )
    # env: gym_wrapper.GymWrapper = gym_wrapper.VectorGymWrapper(env)
    # env: BraxAutoResetWrapper = wrapper.wrap_for_brax_training(env)
    # automatically convert between jax ndarrays and torch tensors:
    # env = torch_wrapper.TorchWrapper(env, device=cfg.device)

    # automatically convert between jax ndarrays and torch tensors:
    # env = torch_wrapper.TorchWrapper(env, device=cfg.device, rng=rng)

    # jit_reset = jax.jit(env.reset)
    # jit_step = jax.jit(env.step)

    # Environment warmup
    rng = jax.random.PRNGKey(0)
    observation = env.reset(rng)

    # action = jp.zeros(env.action_size)
    action = jp.zeros(env.action_size)
    # action = torch.zeros(env.action_size).to(cfg.device)
    # env.step(observation, action)
    # Training metrics
    total_steps = 0
    start_time = time.time()
    rng = jax.random.PRNGKey(0)
    state = env.reset(rng)

    # For evaluation
    eval_env = envs.create(
        cfg.env_name, batch_size=1, episode_length=cfg.episode_length
    )
    eval_env = wrapper.brax_training.EpisodeWrapper(
        eval_env, cfg.episode_length, action_repeat=1
    )

    # Training metrics
    start_time = time.time()

    eval_interval = cfg.num_timesteps // cfg.num_evals
    eval_frequency = cfg.num_timesteps // cfg.num_evals
    next_eval_step = eval_interval  # First evaluation point

    print("ready?")
    input()
    while total_steps < cfg.num_timesteps:
        # Collect experience
        transitions = []

        for _ in range(cfg.unroll_length):
            # Convert JAX observation to PyTorch tensor
            obs_torch = _jax_to_torch(state.obs).to(cfg.device)

            # Get action from agent
            action, log_prob, value = agent.get_action(obs_torch)
            # action, log_prob, value = agent.get_action(obs_torch)

            # Convert action back to JAX array
            action_jax = _torch_to_jax(torch.tensor(action))

            # Step environment
            next_state = env.step(state, action_jax)

            # Store transition
            transitions.append(
                StepData(
                    obs=state.obs,
                    action=action,
                    reward=next_state.reward,
                    next_obs=next_state.obs,
                    done=next_state.done,
                    log_prob=log_prob,
                    value=value,
                )
            )

            state = next_state

        # Update observation normalization
        obs_batch = np.stack([t.obs for t in transitions])
        agent.update_normalization(obs_batch)

        # Convert transitions to batch
        batch = StepData(*zip(*transitions))

        # Update policy
        metrics = agent.update(batch)

        # Update total steps
        total_steps += cfg.unroll_length * cfg.num_envs

        # Evaluation
        if total_steps % eval_frequency == 0:
            eval_rng = jax.random.PRNGKey(0)
            eval_state = eval_env.reset(eval_rng)
            eval_rewards = []

            for _ in range(cfg.episode_length):
                obs_torch = _jax_to_torch(eval_state.obs).to(cfg.device)
                action, _, _ = agent.get_action(obs_torch)
                action_jax = _torch_to_jax(torch.tensor(action))
                eval_state = eval_env.step(eval_state, action_jax)
                eval_rewards.append(eval_state.reward)

            avg_reward = np.mean(eval_rewards)

            # Calculate metrics
            steps_per_sec = total_steps / (time.time() - start_time)

            if progress_cb:
                progress_metrics = {
                    "timesteps": total_steps,
                    "eval/episode_reward": avg_reward,
                    "speed/steps_per_sec": steps_per_sec,
                    **metrics,
                }
                progress_cb(total_steps, progress_metrics)

    return agent

    # while total_steps < cfg.num_timesteps:
    #     # Training phase
    #     train_start_time = time.time()

    #     # Collect experience
    #     observation, td = train_unroll(
    #         agent,
    #         env,
    #         observation,
    #         num_unrolls=cfg.batch_size * cfg.num_minibatches // cfg.num_envs,
    #         unroll_length=cfg.unroll_length,
    #     )

    #     # Process trajectory data
    #     def unroll_first(data):
    #         data = data.swapaxes(0, 1)
    #         return data.reshape([data.shape[0], -1] + list(data.shape[3:]))

    #     td = sd_map(unroll_first, td)

    #     # Update observation normalization if needed
    #     agent.update_normalization(td.observation)

    #     # Calculate steps collected in this batch
    #     steps_collected = cfg.batch_size * cfg.num_minibatches * cfg.unroll_length
    #     total_steps += steps_collected

    #     # Update policy
    #     epoch_losses = []
    #     for _ in range(cfg.num_update_epochs):
    #         # Shuffle and batch the data
    #         with torch.no_grad():
    #             permutation = torch.randperm(td.observation.shape[1], device=cfg.device)

    #             def shuffle_batch(data):
    #                 data = data[:, permutation]

    #                 # Calculate divisible chunk size
    #                 chunk_size = (
    #                     data.shape[1] // cfg.num_minibatches
    #                 ) * cfg.num_minibatches
    #                 data = data[:, :chunk_size]  # Trim to divisible size

    #                 data = data.reshape(
    #                     [data.shape[0], cfg.num_minibatches, -1] + list(data.shape[2:])
    #                 )
    #                 return data.swapaxes(0, 1)

    #             # def shuffle_batch(data):
    #             #     data = data[:, permutation]
    #             #     data = data.reshape(
    #             #         [data.shape[0], cfg.num_minibatches, -1] + list(data.shape[2:])
    #             #     )
    #             #     return data.swapaxes(0, 1)

    #             epoch_td = sd_map(shuffle_batch, td)

    #         # Minibatch updates
    #         for minibatch_i in range(cfg.num_minibatches):
    #             td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
    #             loss = agent.loss(td_minibatch._asdict())
    #             # loss = agent.update(td_minibatch._asdict())
    #             epoch_losses.append(loss.item())

    #     # Calculate metrics
    #     train_time = time.time() - train_start_time
    #     sps = steps_collected / train_time
    #     avg_loss = np.mean(epoch_losses)

    #     # Evaluation phase
    #     # if steps_collected % eval_frequency == 0:
    #     if total_steps >= next_eval_step:
    #         # if total_steps % eval_frequency == 0:
    #         # if (total_steps // steps_collected) % eval_frequency == 0:
    #         eval_start_time = time.time()
    #         with torch.no_grad():
    #             episodes, episode_reward = eval_unroll(agent, env, cfg.episode_length)
    #         eval_time = time.time() - eval_start_time

    #         if progress_cb:
    #             progress = {
    #                 "timesteps": total_steps,
    #                 "eval/episode_reward": episode_reward.item(),
    #                 "eval/completed_episodes": episodes.item(),
    #                 "speed/sps": sps,
    #                 "speed/eval_sps": (cfg.num_envs * cfg.episode_length) / eval_time,
    #                 "losses/total_loss": avg_loss,
    #             }
    #             progress_cb(total_steps, progress)
    #         next_eval_step += eval_interval

    # return agent


if __name__ == "__main__":
    train()
