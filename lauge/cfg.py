import dataclasses
import json
from abc import ABC
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Callable, Type, TypeVar, Union

import yaml  # pip install pyyaml
from brax.training.agents.ppo import train as ppo

T = TypeVar("T", bound="Cfg")


@dataclass
class Cfg:
    """
    Configuration arguments for training and environment setup.
    """

    env_name: str
    """Name of the environment to train on."""

    algo: Callable = ppo.train
    """Algorithm to use for training (e.g., 'ppo.train', 'sac.train')."""

    num_timesteps: int = 20_000_000
    """Total number of timesteps to train the agent."""

    num_evals: int = 5
    """Number of evaluations to perform during training."""

    reward_scaling: float = 0.1
    """Scaling factor for rewards."""

    episode_length: int = 1000
    """Maximum length of an episode in steps."""

    normalize_observations: bool = True
    """Whether to normalize observations during training."""

    action_repeat: int = 1
    """Number of times each action is repeated."""

    unroll_length: int = 10
    """Length of unrolled sequences during training."""

    num_minibatches: int = 24
    """Number of minibatches for policy updates."""

    num_updates_per_batch: int = 8
    """Number of updates per batch of data."""

    discounting: float = 0.97
    """Discount factor for future rewards."""

    learning_rate: float = 3e-4
    """Learning rate for the optimizer."""

    entropy_cost: float = 1e-3
    """Coefficient for the entropy regularization term."""

    num_envs: int = 3072
    """Number of parallel environments to run during training."""

    batch_size: int = 512
    """Batch size used during training."""

    seed: int = 0
    """Random seed for reproducibility."""

    backend: str = "mjx"
    """brax backend to use in simulation (default: 'mjx')."""

    device: str = "cuda"
    """Hardware accelerator to use for simulation (e.g., 'cuda', 'cpu'). """

    policy_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    """Layer sizes for policy network (default: [64, 64]). 
    
    Each integer represents the number of units in a hidden layer. The network will have
    this many hidden layers with the specified sizes.
    """

    value_hidden_layer_sizes: tuple[int, ...] = (64, 64)
    """Layer sizes for value function network (default: [64, 64]).
    
    Each integer represents the number of units in a hidden layer. The network will have
    this many hidden layers with the specified sizes. Typically matches the policy network
    architecture unless specifically designing asymmetric networks.
    """

    num_update_epochs: int = 4
    """Number of passes over collected data during each training iteration (default: 4)."""

    def __post_init__(self):
        """Auto-initialize nested configs"""
        self._init_nested()

    def _init_nested(self):
        """Initialize nested config objects that are None"""
        for field_name, field_type in self.__annotations__.items():
            if hasattr(field_type, "_init_nested"):  # Check if it's a config class
                if getattr(self, field_name) is None:
                    default_value = field_type()
                    setattr(self, field_name, default_value)

    def to_dict(self) -> dict:
        """Convert config to dict, handling nested configs"""

        def _convert(value: Any) -> Any:
            if is_dataclass(value):
                return {k: _convert(v) for k, v in asdict(value).items()}
            elif isinstance(value, (list, tuple)):
                return [_convert(v) for v in value]
            elif isinstance(value, dict):
                return {k: _convert(v) for k, v in value.items()}
            return value

        return _convert(asdict(self))

    def to_json(self, file_path: str = None) -> str:
        """Serialize to JSON string or file"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if file_path:
            with open(file_path, "w") as f:
                f.write(json_str)
        return json_str

    def to_yaml(self, file_path: str = None) -> str:
        """Serialize to YAML string or file"""
        yaml_str = yaml.safe_dump(self.to_dict(), sort_keys=False)
        if file_path:
            with open(file_path, "w") as f:
                f.write(yaml_str)
        return yaml_str

    @classmethod
    def from_dict(cls: Type[T], data: dict) -> T:
        """Create config from dict (recursively)"""

        def _parse(_cls: Type[T], _data: dict) -> T:
            if is_dataclass(_cls):
                field_types = {f.name: f.type for f in dataclasses.fields(_cls)}
                kwargs = {}
                for name, value in _data.items():
                    if name in field_types:
                        field_type = field_types[name]
                        if is_dataclass(field_type):
                            kwargs[name] = _parse(field_type, value)
                        else:
                            kwargs[name] = value
                return _cls(**kwargs)
            return _data

        return _parse(cls, data)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Create config from JSON string"""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls: Type[T], file_path: str) -> T:
        """Create config from JSON file"""
        with open(file_path) as f:
            return cls.from_dict(json.load(f))

    @classmethod
    def from_yaml(cls: Type[T], yaml_str: str) -> T:
        """Create config from YAML string"""
        return cls.from_dict(yaml.safe_load(yaml_str))

    @classmethod
    def from_yaml_file(cls: Type[T], file_path: str) -> T:
        """Create config from YAML file"""
        with open(file_path) as f:
            return cls.from_dict(yaml.safe_load(f))

    # def __post_init__(self):
    #     self._init_nested()
    #     self._validate()

    # def _validate(self):
    #     """Add validation logic for fields"""
    #     if hasattr(self, "port") and not (0 < self.port <= 65535):
    #         raise ValueError("Invalid port number")
    #     # Add other validation rules


@dataclass
class TD3Cfg(Cfg):
    num_timesteps: int = 2_000_000
    num_evals: int = 20
    reward_scaling: int = 1
    episode_length: int = 1000
    normalize_observations: bool = True
    discounting: int = 0.97
    learning_rate: int = 3e-4
    adam_eps: int = 1e-5
    num_envs: int = 128
    num_eval_envs: int = 128
    batch_size: int = 64
    grad_updates_per_step: int = 50
    alpha: float = 2.5
    max_replay_size: int = 1_000_000
    min_replay_size: int = 8192
    policy_hidden_layer_sizes: tuple[int, ...] = (256, 256)
    value_hidden_layer_sizes: tuple[int, ...] = (256, 256)
    noise_clip: float = 0.5
    tau: float = 0.005
    policy_noise: float = 0.2
    policy_delay: float = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4


@dataclass
class PPOCfg(Cfg): ...


@dataclass(kw_only=True)
class IBRLCfg(Cfg):
    rl_config: Cfg
    il_config: Cfg
    # env_name: str = "<placeholder>"
    soft_update_beta: float = 0.2
    policy_delay: int = 2
    smooth_noise: float = 0.1
    smooth_clip: float = 0.1
    polyak: float = 0.005


def resolve_param(batch_size=None, num_minibatches=None, num_envs=None):
    """
    Compute the missing parameter such that batch_size * num_minibatches % num_envs == 0.
    At least two parameters must be provided, and the third one will be computed.

    Args:
        batch_size (int, optional): The size of each batch.
        num_minibatches (int, optional): The number of minibatches.
        num_envs (int, optional): The number of environments.

    Returns:
        int: The computed value of the missing parameter.

    Raises:
        ValueError: If more than one parameter is missing or the inputs are invalid.
    """
    if sum(x is None for x in (batch_size, num_minibatches, num_envs)) != 1:
        raise ValueError("Exactly one parameter must be None.")

    if num_envs is None:
        if batch_size is None or num_minibatches is None:
            raise ValueError("batch_size and num_minibatches must be provided.")
        num_envs = batch_size * num_minibatches
        return num_envs

    if batch_size is None:
        if num_minibatches is None or num_envs is None:
            raise ValueError("num_minibatches and num_envs must be provided.")
        if num_envs % num_minibatches != 0:
            raise ValueError("No valid batch_size can satisfy the constraint.")
        batch_size = num_envs // num_minibatches
        return batch_size

    if num_minibatches is None:
        if batch_size is None or num_envs is None:
            raise ValueError("batch_size and num_envs must be provided.")
        if num_envs % batch_size != 0:
            raise ValueError("No valid num_minibatches can satisfy the constraint.")
        num_minibatches = num_envs // batch_size
        return num_minibatches



