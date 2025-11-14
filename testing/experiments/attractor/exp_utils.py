# import isaacgym
# import isaacgymenvs
import glob
import json
import os
import pickle
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import tqdm

# import the skrl components to build the RL system
from skrl.envs.torch import wrap_env

# from algorithms.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory

# Import the skrl components to build the RL system
from skrl.models.torch import DeterministicMixin, Model

# Import the skrl components to build the RL system
from skrl.resources.noises.torch import GaussianNoise
from skrl.trainers.torch import SequentialTrainer

# from skrl.trainers.torch import SequentialTrainer
# from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG
from skrl.utils import set_seed
from testing import wrappers as wrap
from testing.envs.xpose import XPose

# from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from testing.shen.BC import BC, BC_DEFAULT_CONFIG

# from algorithms.IBRL_active import IBRL
# from testing.shen.ibrl import IBRL, IBRL_DEFAULT_CONFIG
# from testing.shen.IBRL import IBRL, IBRL_DEFAULT_CONFIG
from testing.shen.drlr import DRLR_DEFAULT_CONFIG
from testing.shen.ibrl_rl import IBRL_RL_DEFAULT_CONFIG
from testing.train.demon import TransitionDataset

_SEED = 10


class DeterministicActor(DeterministicMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        dropout_rate=0.1,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.num_actions),
        )

    def compute(self, inputs, role):
        raw_action = self.net(inputs["states"])
        return torch.tanh(raw_action), {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 256)
        self.linear_layer_2 = nn.Linear(256, 256)
        self.linear_layer_3 = nn.Linear(256, 1)

    def compute(self, inputs, role):
        x = F.relu(
            self.linear_layer_1(
                torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
            )
        )
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}


class BCmodel(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, self.num_actions),
        )

    def compute(self, inputs, role):
        return self.net(inputs["states"]), {}


def setup_environment(
    batch_size: int = 100,
    episode_length: int = 1000,
    auto_reset: bool = True,
    action_repeat: int = 1,
):
    """Set up the MJX XPose environment with proper wrapping."""

    # Create base environment
    # env = cartpole.Balance(swing_up=False, sparse=False)
    # env = pendulum.SwingUp()
    env = XPose()

    env = wrap.create(
        env,
        batch_size=batch_size,
        episode_length=episode_length,
        auto_reset=auto_reset,
        action_repeat=action_repeat,
    )
    env = wrap_env(env, wrapper="playground")

    return env


def get_expert_memory(expert_data_dir: str = "data/norm_data") -> RandomMemory:
    # data_path = Path("data/raw")
    # data_path = Path("data/norm_smooth_states_and_actions")
    # data_path = Path("data/norm_data")
    data_path = Path(
        expert_data_dir
        # "./data/norm_smooth_data_test_2/"
    )  # data/norm_smooth_states_and_actions
    data_files = list(data_path.glob("*.json"))

    def reward_fn(state, action, next_state):
        return -np.linalg.norm(state)  # Negative distance as reward

    def termination_fn(state, next_state):
        return np.linalg.norm(state) < 0.0001

    dataset = TransitionDataset(
        json_paths=data_files, reward_fn=reward_fn, termination_fn=termination_fn
    )

    # Get all transitions from the dataset
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []

    for i in range(len(dataset)):
        transition = dataset[i]
        states.append(transition["state"].numpy())
        actions.append(transition["action"].numpy())

        if "next_state" in transition:
            next_states.append(transition["next_state"].numpy())
        else:
            # Handle case where next_state might not be available
            next_states.append(transition["state"].numpy())  # Fallback

        if "reward" in transition:
            rewards.append(transition["reward"].numpy())
        else:
            rewards.append(0.0)  # Default reward

        if "done" in transition:
            dones.append(transition["done"].numpy())
        else:
            dones.append(False)  # Default not done

    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    terminated = torch.tensor(dones, dtype=torch.float32)

    memory_size = len(states)

    a_dim = actions.shape[1]  # Get action dimension from data
    o_dim = states.shape[1]  # Get observation dimension from data

    # Create expert memory
    expert_memory = RandomMemory(memory_size=memory_size)
    expert_memory.create_tensor(name="states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="actions", size=a_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="next_states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
    expert_memory.create_tensor(name="terminated", size=1, dtype=torch.float32)

    # Add samples to memory
    expert_memory.add_samples(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards.unsqueeze(-1),  # Add dimension for reward size
        terminated=terminated.unsqueeze(-1),  # Add dimension for terminated size
    )
    return expert_memory


def get_memory(env) -> RandomMemory:
    memory = RandomMemory(
        memory_size=350000, num_envs=env.num_envs, device=env.device, replacement=True
    )
    return memory


def generate_expert_memory() -> None:
    data_path = Path("./data/norm_smooth_data/")
    data_files = list(data_path.glob("*.json"))

    def reward_fn(state, action, next_state):
        return -np.linalg.norm(state)  # Negative distance as reward

    def termination_fn(state, next_state):
        return np.linalg.norm(state) < 0.0001

    dataset = TransitionDataset(
        json_paths=data_files, reward_fn=reward_fn, termination_fn=termination_fn
    )

    # Get all transitions from the dataset
    states = []
    actions = []
    next_states = []
    rewards = []
    dones = []

    for i in range(len(dataset)):
        transition = dataset[i]
        states.append(transition["state"].numpy())
        actions.append(transition["action"].numpy())

        if "next_state" in transition:
            next_states.append(transition["next_state"].numpy())
        else:
            # Handle case where next_state might not be available
            next_states.append(transition["state"].numpy())  # Fallback

        if "reward" in transition:
            rewards.append(transition["reward"].numpy())
        else:
            rewards.append(0.0)  # Default reward

        if "done" in transition:
            dones.append(transition["done"].numpy())
        else:
            dones.append(False)  # Default not done

    # Convert to arrays
    states = np.array(states)
    actions = np.array(actions)
    next_states = np.array(next_states)
    rewards = np.array(rewards)
    dones = np.array(dones)

    # Convert to tensors
    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    next_states = torch.tensor(next_states, dtype=torch.float32)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    terminated = torch.tensor(dones, dtype=torch.float32)

    memory_size = len(states)

    a_dim = actions.shape[1]  # Get action dimension from data
    o_dim = states.shape[1]  # Get observation dimension from data

    # Create expert memory
    expert_memory = RandomMemory(memory_size=memory_size)
    expert_memory.create_tensor(name="states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="actions", size=a_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="next_states", size=o_dim, dtype=torch.float32)
    expert_memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
    expert_memory.create_tensor(name="terminated", size=1, dtype=torch.float32)

    # Add samples to memory
    expert_memory.add_samples(
        states=states,
        actions=actions,
        next_states=next_states,
        rewards=rewards.unsqueeze(-1),  # Add dimension for reward size
        terminated=terminated.unsqueeze(-1),  # Add dimension for terminated size
    )

    file_name = "memories/expert_memory.pkl"

    with open(file_name, "wb") as f:
        pickle.dump(expert_memory, f)

    print(f"Saved memory to {file_name}")


def get_td3_models(env) -> dict:
    device = env.device
    models_td3 = {}
    models_td3["policy"] = DeterministicActor(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    models_td3["target_policy"] = DeterministicActor(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    models_td3["critic_1"] = Critic(env.observation_space, env.action_space, device)
    models_td3["critic_2"] = Critic(env.observation_space, env.action_space, device)
    models_td3["target_critic_1"] = Critic(
        env.observation_space, env.action_space, device
    )
    models_td3["target_critic_2"] = Critic(
        env.observation_space, env.action_space, device
    )

    # Initialize the models' parameters (weights and biases) using a Gaussian distribution
    for model in models_td3.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return models_td3


def get_ibrl_config(exp_name: str, env) -> dict:
    device = env.device
    cfg_IBRL = IBRL_RL_DEFAULT_CONFIG.copy()
    cfg_IBRL["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    # cfg_IBRL["exploration"]["noise"] = None
    cfg_IBRL["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
    cfg_IBRL["smooth_regularization_clip"] = 0.5
    cfg_IBRL["gradient_steps"] = 1
    cfg_IBRL["RED-Q_enable"] = False
    # cfg_IBRL["RED-Q_enable"] = True
    cfg_IBRL["offline"] = False
    # cfg_IBRL["offline"] = True
    cfg_IBRL["batch_size"] = 128
    cfg_IBRL["random_timesteps"] = 0
    cfg_IBRL["learning_starts"] = 0
    cfg_IBRL["learning_rate"] = 3e-4
    cfg_IBRL["num_envs"] = env.num_envs
    # cfg_IBRL["demo_file"] = "/home/chen/Downloads/new/memories/Cab-expert-bc.csv"
    cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_IBRL["experiment"]["write_interval"] = 500
    cfg_IBRL["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    # cfg_IBRL["experiment"]["write_interval"] = 50
    # cfg_IBRL["experiment"]["checkpoint_interval"] = 100
    cfg_IBRL["experiment"]["directory"] = model_path.as_posix()
    cfg_IBRL["experiment"]["experiment_name"] = exp_name
    cfg_IBRL["experiment"]["wandb"] = True
    return cfg_IBRL


def get_bc_models(env) -> dict:
    device = env.device
    models_BC = {}
    models_BC["policy"] = BCmodel(
        env.observation_space, env.action_space, device, clip_actions=True
    )
    for model in models_BC.values():
        model.init_parameters(method_name="normal_", mean=0.0, std=0.1)
    return models_BC


def get_drlr_config(exp_name: str, env, wandb: bool = True) -> dict:
    device = env.device
    cfg_IBRL = DRLR_DEFAULT_CONFIG.copy()
    cfg_IBRL["exploration"]["noise"] = GaussianNoise(0, 0.1, device=device)
    # cfg_IBRL["exploration"]["noise"] = None
    cfg_IBRL["smooth_regularization_noise"] = GaussianNoise(0, 0.1, device=device)
    cfg_IBRL["smooth_regularization_clip"] = 0.5
    cfg_IBRL["gradient_steps"] = 1
    cfg_IBRL["RED-Q_enable"] = False
    # cfg_IBRL["RED-Q_enable"] = True
    cfg_IBRL["offline"] = False
    # cfg_IBRL["offline"] = True
    cfg_IBRL["batch_size"] = 128
    cfg_IBRL["random_timesteps"] = 0
    cfg_IBRL["learning_starts"] = 0
    cfg_IBRL["learning_rate"] = 3e-4
    cfg_IBRL["num_envs"] = env.num_envs
    # cfg_IBRL["demo_file"] = "/home/chen/Downloads/new/memories/Cab-expert-bc.csv"
    cfg_IBRL["demo_file"] = "./Demos/cab_imperfect.csv"
    # logging to TensorBoard and write checkpoints each 25 and 1000 timesteps respectively
    cfg_IBRL["experiment"]["write_interval"] = 100
    # cfg_IBRL["experiment"]["write_interval"] = 500
    cfg_IBRL["experiment"]["checkpoint_interval"] = 1000

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg_IBRL["experiment"]["directory"] = model_path.as_posix()
    cfg_IBRL["experiment"]["experiment_name"] = exp_name
    cfg_IBRL["experiment"]["wandb"] = wandb
    return cfg_IBRL


def get_bc_config(exp_name: str, env, wandb: bool = True) -> dict:
    device = env.device
    cfg_BC = BC_DEFAULT_CONFIG.copy()
    cfg_BC["gradient_steps"] = 5
    cfg_BC["batch_size"] = 256
    cfg_BC["demo_file"] = "./Demos/Cab-expert-bc.csv"
    cfg_BC["exploration"]["noise"] = GaussianNoise(0, 0.0001, device=device)
    cfg_BC["smooth_regularization_clip"] = 0.0001

    # Experiment configuration
    model_path = Path(__file__).parent / "results/models"
    model_path.mkdir(parents=True, exist_ok=True)

    cfg_BC["experiment"]["write_interval"] = 50
    cfg_BC["experiment"]["checkpoint_interval"] = 100
    cfg_BC["experiment"]["directory"] = model_path.as_posix()
    cfg_BC["experiment"]["experiment_name"] = exp_name
    cfg_BC["experiment"]["wandb"] = wandb
    return cfg_BC


def get_trainer(env, agent, timesteps: int = 350_000) -> SequentialTrainer:
    cfg = {"timesteps": timesteps, "headless": True}
    trainer = SequentialTrainer(cfg=cfg, env=env, agents=agent)
    return trainer


def exp_set_seed():
    set_seed(_SEED)


def rollout(
    file_name: str,
    env,
    agent,
    num_timesteps: int = 1000,
    end_on_terminate: bool = False,
):
    agent.set_mode("eval")
    state, _ = env.reset()
    frames = []

    data = {"states": [], "actions": []}

    print("Performing rollout")
    for i in tqdm(range(num_timesteps)):
        actions, _, _ = agent.act(states=state, timestep=i, timesteps=num_timesteps)
        data["actions"].append(actions.tolist())
        data["states"].append(state.tolist())
        next_states, rewards, terminated, truncated, infos = env.step(
            actions=actions.detach()
        )
        frame = env.render()
        frames.append(frame)
        state = next_states
        if terminated[0]:
            if end_on_terminate:
                break
            state, _ = env.reset()

    with open("data/tmp/data.json", "w") as f:
        json.dump(data, f, indent=4)

    # Create video from frames
    if frames:
        # Get frame dimensions
        height, width = frames[0].shape[:2]

        # Use pathlib for proper path handling
        output_path = Path(file_name)

        # Create temp file in the same directory with a proper name
        temp_file_name = output_path.parent / f"temp_{output_path.name}"

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Writing {len(frames)} frames to temporary file: {temp_file_name}")

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = 30

        out = cv2.VideoWriter(str(temp_file_name), fourcc, fps, (width, height))

        # Write all frames to video
        for frame in frames:
            # Ensure frame is in correct format (BGR for OpenCV)
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # Convert RGB to BGR if needed
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)

        out.release()
        print(f"Temporary video saved: {temp_file_name}")

        # Check if temp file was created successfully
        if not temp_file_name.exists():
            print(f"Error: Temporary file was not created: {temp_file_name}")
            return

        # Convert to H.264 using FFmpeg
        print("Converting to H.264 format...")
        try:
            # Run FFmpeg conversion
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",  # -y to overwrite output file
                    "-i",
                    str(temp_file_name),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            print(f"Successfully converted to H.264: {output_path}")

            # Remove temporary file
            temp_file_name.unlink()
            print(f"Removed temporary file: {temp_file_name}")

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg conversion failed: {e}")
            print(f"FFmpeg stderr: {e.stderr}")

            # Check if temp file exists and has content
            if temp_file_name.exists():
                file_size = temp_file_name.stat().st_size
                print(f"Temporary file exists with size: {file_size} bytes")

                # Try to use the temp file directly
                try:
                    temp_file_name.rename(output_path)
                    print(f"Renamed temporary file to: {output_path}")
                except Exception as rename_error:
                    print(f"Failed to rename temporary file: {rename_error}")
            else:
                print("Temporary file does not exist")

        except FileNotFoundError:
            print("FFmpeg not found. Please install ffmpeg.")
            # Try to rename the temp file
            try:
                temp_file_name.rename(output_path)
                print(f"Renamed temporary file to: {output_path}")
            except Exception as rename_error:
                print(f"Failed to rename temporary file: {rename_error}")

    else:
        print("No frames were generated")


def create_data(
    input_dir: str, output_dir: str = "data/test", state_filter: bool = False
):
    def low_pass_filter(data, window_size=5):
        """Apply simple moving average filter"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size) / window_size, mode="same")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the data directory
    path = Path(input_dir)
    data_files = glob.glob((path / "*.json").as_posix())

    # Define colors for different files
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_files)))

    for i, file_path in enumerate(data_files):
        with open(file_path) as f:
            data = json.load(f)

        # Original data
        states_original = np.array(data["states"])
        actions_original = np.array(data["actions"])

        s_x_orig = states_original[:, 0]
        s_y_orig = states_original[:, 1]
        s_z_orig = states_original[:, 2]
        a_x_orig = actions_original[:, 0]
        a_y_orig = actions_original[:, 1]
        a_z_orig = actions_original[:, 2]

        # Apply low-pass filter to create filtered data
        window_size = 100

        if state_filter:
            s_x_filt = low_pass_filter(s_x_orig, window_size)
            s_y_filt = low_pass_filter(s_y_orig, window_size)
            s_z_filt = low_pass_filter(s_z_orig, window_size)
        else:
            s_x_filt = s_x_orig
            s_y_filt = s_y_orig
            s_z_filt = s_z_orig

        a_x_filt = low_pass_filter(a_x_orig, window_size)
        a_y_filt = low_pass_filter(a_y_orig, window_size)
        a_z_filt = low_pass_filter(a_z_orig, window_size)

        # Create new actions array with filtered data
        actions_filtered = np.column_stack((a_x_filt, a_y_filt, a_z_filt))
        states_filtered = np.column_stack((s_x_filt, s_y_filt, s_z_filt))

        # Create new data dictionary with filtered actions
        data_filtered = data.copy()
        data_filtered["actions"] = actions_filtered.tolist()
        data_filtered["states"] = states_filtered.tolist()

        # Generate output filename
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)
        print(output_path)

        # Save filtered data
        with open(output_path, "w") as f:
            json.dump(data_filtered, f, indent=2)

        print(f"Saved data to {output_dir}")

        # Get filename for legend (without extension)
        label = os.path.splitext(filename)[0]

        # Create comparison figure
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(f"Data Comparison: {label}", fontsize=16, fontweight="bold")

        # Plot states comparison
        time_original = np.arange(len(states_original))
        time_filtered = np.arange(len(states_filtered))

        # State X
        axes[0, 0].plot(
            time_original, s_x_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[0, 0].plot(time_filtered, s_x_filt, "r-", label="Filtered", linewidth=2)
        axes[0, 0].set_ylabel("State X")
        axes[0, 0].set_title("State X - Before vs After Filtering")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # State Y
        axes[1, 0].plot(
            time_original, s_y_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[1, 0].plot(time_filtered, s_y_filt, "r-", label="Filtered", linewidth=2)
        axes[1, 0].set_ylabel("State Y")
        axes[1, 0].set_title("State Y - Before vs After Filtering")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # State Z
        axes[2, 0].plot(
            time_original, s_z_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[2, 0].plot(time_filtered, s_z_filt, "r-", label="Filtered", linewidth=2)
        axes[2, 0].set_ylabel("State Z")
        axes[2, 0].set_xlabel("Time Steps")
        axes[2, 0].set_title("State Z - Before vs After Filtering")
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)

        # Plot actions comparison
        # Action X
        axes[0, 1].plot(
            time_original, a_x_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[0, 1].plot(time_filtered, a_x_filt, "r-", label="Filtered", linewidth=2)
        axes[0, 1].set_ylabel("Action X")
        axes[0, 1].set_title("Action X - Before vs After Filtering")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Action Y
        axes[1, 1].plot(
            time_original, a_y_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[1, 1].plot(time_filtered, a_y_filt, "r-", label="Filtered", linewidth=2)
        axes[1, 1].set_ylabel("Action Y")
        axes[1, 1].set_title("Action Y - Before vs After Filtering")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Action Z
        axes[2, 1].plot(
            time_original, a_z_orig, "b-", alpha=0.7, label="Original", linewidth=1
        )
        axes[2, 1].plot(time_filtered, a_z_filt, "r-", label="Filtered", linewidth=2)
        axes[2, 1].set_ylabel("Action Z")
        axes[2, 1].set_xlabel("Time Steps")
        axes[2, 1].set_title("Action Z - Before vs After Filtering")
        axes[2, 1].legend()
        axes[2, 1].grid(True, alpha=0.3)

        # Adjust layout
        plt.tight_layout()

        # Save the figure
        figure_filename = f"{label}_comparison.png"
        figure_path = os.path.join(output_dir, figure_filename)
        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        print(f"Saved comparison figure to {figure_path}")

        # Close the figure to free memory
        plt.close(fig)

    print("Processing completed!")
