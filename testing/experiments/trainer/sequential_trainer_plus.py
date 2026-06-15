import copy
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import tqdm

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer
from skrl.trainers.torch.sequential import SEQUENTIAL_TRAINER_DEFAULT_CONFIG


class SequentialTrainerPlus(Trainer):
    """Sequential trainer that forwards terminated flags to agents' post_interaction.

    This mirrors skrl.trainers.torch.sequential.SequentialTrainer but ensures
    post_interaction receives the terminated tensor for agents (e.g. IBRL) that
    expect it.
    """

    def __init__(
        self,
        env: Wrapper,
        agents: Union[Agent, List[Agent]],
        agents_scope: Optional[List[int]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(SEQUENTIAL_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(env=env, agents=agents, agents_scope=agents_scope, cfg=_cfg)

        self.rollout_video_every_episodes = self.cfg.get(
            "rollout_video_every_episodes", 0
        )
        self.rollout_video_num_steps = self.cfg.get("rollout_video_num_steps", 1000)
        self.rollout_video_fps = self.cfg.get("rollout_video_fps", 30)
        self.rollout_video_dir = self.cfg.get("rollout_video_dir")
        self.rollout_video_prefix = self.cfg.get("rollout_video_prefix", "rollout")
        self.rollout_video_env_index = int(
            self.cfg.get("rollout_video_env_index", 0) or 0
        )
        self.rollout_video_count_all_envs = bool(
            self.cfg.get("rollout_video_count_all_envs", False)
        )
        self.rollout_video_disable_tracking = bool(
            self.cfg.get("rollout_video_disable_tracking", True)
        )
        self.eval_env = self.cfg.get("eval_env")
        self._episode_count = 0
        self._next_rollout_episode = (
            self.rollout_video_every_episodes
            if self.rollout_video_every_episodes
            else None
        )
        self.log_rollout_path = self.cfg.get("log_rollout_path")
        self.log_rollout_steps = int(self.cfg.get("log_rollout_steps", 0) or 0)
        self.log_rollout_exit = bool(self.cfg.get("log_rollout_exit", False))
        self._rollout_log = {"states": [], "actions": []}
        self._rollout_log_saved = False

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def _iter_agents(self):
        if self.num_simultaneous_agents > 1:
            return list(self.agents)
        return [self.agents]

    def _suspend_agent_logging(self):
        if not self.rollout_video_disable_tracking:
            return []
        saved = []
        for agent in self._iter_agents():
            saved.append(
                {
                    "agent": agent,
                    "write_interval": getattr(agent, "write_interval", None),
                    "checkpoint_interval": getattr(agent, "checkpoint_interval", None),
                    "track_data": getattr(agent, "track_data", None),
                    "write_tracking_data": getattr(agent, "write_tracking_data", None),
                }
            )
            if hasattr(agent, "write_interval"):
                agent.write_interval = 0
            if hasattr(agent, "checkpoint_interval"):
                agent.checkpoint_interval = 0
            if hasattr(agent, "track_data"):
                agent.track_data = lambda *args, **kwargs: None
            if hasattr(agent, "write_tracking_data"):
                agent.write_tracking_data = lambda *args, **kwargs: None
        return saved

    @staticmethod
    def _restore_agent_logging(saved):
        for entry in saved:
            agent = entry["agent"]
            if entry["write_interval"] is not None:
                agent.write_interval = entry["write_interval"]
            if entry["checkpoint_interval"] is not None:
                agent.checkpoint_interval = entry["checkpoint_interval"]
            if entry["track_data"] is not None:
                agent.track_data = entry["track_data"]
            if entry["write_tracking_data"] is not None:
                agent.write_tracking_data = entry["write_tracking_data"]

    def train(self) -> None:
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents reuse overridden helpers
        if self.num_simultaneous_agents == 1:
            if self.env.num_agents == 1:
                self.single_agent_train()
            else:
                self.multi_agent_train()
            return

        # reset env
        states, infos = self.env.reset()
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = torch.vstack(
                    [
                        agent.act(
                            states[scope[0] : scope[1]],
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )[0]
                        for agent, scope in zip(self.agents, self.agents_scope)
                    ]
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )
                self._maybe_log_rollout(
                    states=states, actions=actions, timestep=timestep
                )

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction (pass terminated for agent-specific handling)
            for agent in self.agents:
                agent.post_interaction(
                    terminated=terminated, timestep=timestep, timesteps=self.timesteps
                )

            # reset environments
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

        self._maybe_save_rollout_log()

    def eval(self) -> None:
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("eval")
        else:
            self.agents.set_running_mode("eval")

        # non-simultaneous agents reuse overridden helpers
        if self.num_simultaneous_agents == 1:
            if self.env.num_agents == 1:
                self.single_agent_eval()
            else:
                self.multi_agent_eval()
            return

        states, infos = self.env.reset()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                outputs = [
                    agent.act(
                        states[scope[0] : scope[1]],
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                    for agent, scope in zip(self.agents, self.agents_scope)
                ]
                actions = torch.vstack(
                    [
                        output[0]
                        if self.stochastic_evaluation
                        else output[-1].get("mean_actions", output[0])
                        for output in outputs
                    ]
                )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                for agent, scope in zip(self.agents, self.agents_scope):
                    agent.record_transition(
                        states=states[scope[0] : scope[1]],
                        actions=actions[scope[0] : scope[1]],
                        rewards=rewards[scope[0] : scope[1]],
                        next_states=next_states[scope[0] : scope[1]],
                        terminated=terminated[scope[0] : scope[1]],
                        truncated=truncated[scope[0] : scope[1]],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction (pass terminated for agent-specific handling)
            for agent in self.agents:
                agent.post_interaction(
                    terminated=terminated, timestep=timestep, timesteps=self.timesteps
                )

            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = self.env.reset()
            else:
                states = next_states

    def single_agent_train(self) -> None:
        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents == 1

        states, infos = self.env.reset()
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            self.agents.pre_interaction(
                states=states, timestep=timestep, timesteps=self.timesteps
            )

            with torch.no_grad():
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps
                )[0]
                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                self._maybe_log_rollout(
                    states=states, actions=actions, timestep=timestep
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            self.agents.post_interaction(
                next_states=next_states, timestep=timestep, timesteps=self.timesteps
            )

            episode_ends = (terminated | truncated).view(-1)
            if self.env.num_envs > 1 and not self.rollout_video_count_all_envs:
                env_index = max(
                    0, min(self.rollout_video_env_index, self.env.num_envs - 1)
                )
                ended_count = int(episode_ends[env_index].item())
            else:
                ended_count = int(episode_ends.sum().item())
            if ended_count:
                if self.rollout_video_every_episodes:
                    print(
                        f"[rollout] ended_count={ended_count} "
                        f"episode_count={self._episode_count} "
                        f"next_rollout_episode={self._next_rollout_episode} "
                        f"every={self.rollout_video_every_episodes} "
                        f"count_all_envs={self.rollout_video_count_all_envs} "
                        f"envs={getattr(self.env, 'num_envs', 1)}"
                    )
                self._episode_count += ended_count
                self._maybe_record_rollout()

            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

        self._maybe_save_rollout_log()

    def single_agent_eval(self) -> None:
        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents == 1

        states, infos = self.env.reset()
        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                outputs = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps
                )
                actions = (
                    outputs[0]
                    if self.stochastic_evaluation
                    else outputs[-1].get("mean_actions", outputs[0])
                )

                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                self._maybe_log_rollout(
                    states=states, actions=actions, timestep=timestep
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            self.agents.post_interaction(
                terminated=terminated, timestep=timestep, timesteps=self.timesteps
            )

            if self.env.num_envs > 1:
                states = next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                else:
                    states = next_states

    def _maybe_record_rollout(self) -> None:
        if not self.rollout_video_every_episodes:
            return
        while (
            self._next_rollout_episode is not None
            and self._episode_count >= self._next_rollout_episode
        ):
            self._record_rollout_video(self._next_rollout_episode)
            self._next_rollout_episode += self.rollout_video_every_episodes

    def _record_rollout_video(self, episode_index: int) -> None:
        try:
            import cv2
        except ImportError:
            return

        rollout_env = self.eval_env or self.env
        # Reset any per-agent state buffers that depend on batch size
        for agent in self._iter_agents():
            if hasattr(agent, "_states"):
                agent._states = None
            if hasattr(agent, "_prev_states"):
                agent._prev_states = None
        num_envs = int(getattr(rollout_env, "num_envs", 1) or 1)
        env_index = 0
        if num_envs > 1:
            env_index = max(0, min(self.rollout_video_env_index, num_envs - 1))

        video_dir = self.rollout_video_dir
        if not video_dir:
            base_dir = getattr(self.agents, "experiment_dir", os.getcwd())
            video_dir = os.path.join(base_dir, "media")
        video_dir = Path(video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)

        # print("setting ibrl to eval")

        self.agents.set_running_mode("eval")
        self.agents.set_mode("eval")

        # if hasattr(self.agents, "IL_policy") and hasattr(self.agents.IL_policy, "eval"):
        #     self.agents.IL_policy.eval()
        # Disable learning updates during rollout video (but keep state queue updates).
        orig_learning_starts = getattr(self.agents, "_learning_starts", None)
        orig_gradient_steps = getattr(self.agents, "_gradient_steps", None)
        if orig_learning_starts is not None:
            self.agents._learning_starts = self.rollout_video_num_steps + 1
        if orig_gradient_steps is not None:
            self.agents._gradient_steps = 0

        frames = []
        rollout_log = {"states": [], "actions": [], "time": []}
        steps = 0
        success = False
        logging_state = self._suspend_agent_logging()
        try:
            with torch.no_grad():
                states, _ = rollout_env.reset()
                for step in tqdm.tqdm(
                    range(self.rollout_video_num_steps),
                    desc="rollout",
                    disable=self.disable_progressbar,
                    file=sys.stdout,
                ):
                    self.agents.pre_interaction(
                        states=states,
                        timestep=step,
                        timesteps=self.rollout_video_num_steps,
                    )
                    actions = self.agents.act(
                        states, timestep=step, timesteps=self.rollout_video_num_steps
                    )[0]
                    states, _, terminated, truncated, _ = rollout_env.step(actions)
                    self.agents.post_interaction(
                        next_states=states,
                        timestep=step,
                        timesteps=self.rollout_video_num_steps,
                    )
                    frame = None
                    try:
                        frame = rollout_env.render(
                            mode="rgb_array", env_index=env_index
                        )
                    except TypeError:
                        frame = rollout_env.render()
                    if isinstance(frame, (list, tuple)):
                        frame = frame[env_index] if frame else None
                    elif isinstance(frame, np.ndarray) and frame.ndim == 4:
                        frame = frame[env_index]
                    if frame is not None:
                        frames.append(np.asarray(frame))
                    if num_envs > 1:
                        state_step = states[env_index]
                        action_step = actions[env_index]
                    else:
                        state_step = states
                        action_step = actions
                        if isinstance(state_step, torch.Tensor) and state_step.ndim > 1:
                            state_step = state_step[0]
                        if (
                            isinstance(action_step, torch.Tensor)
                            and action_step.ndim > 1
                        ):
                            action_step = action_step[0]
                    rollout_log["states"].append(state_step.detach().cpu().tolist())
                    rollout_log["actions"].append(action_step.detach().cpu().tolist())
                    steps = step + 1
                    if num_envs > 1:
                        term_flag = bool(terminated.reshape(-1)[env_index])
                        trunc_flag = bool(truncated.reshape(-1)[env_index])
                        if term_flag or trunc_flag:
                            if (
                                term_flag
                                and not trunc_flag
                                and steps < self.rollout_video_num_steps
                            ):
                                success = True
                            break
                    else:
                        term_flag = bool(terminated.any())
                        trunc_flag = bool(truncated.any())
                        if term_flag or trunc_flag:
                            if (
                                term_flag
                                and not trunc_flag
                                and steps < self.rollout_video_num_steps
                            ):
                                success = True
                            break
        finally:
            self._restore_agent_logging(logging_state)

        self.agents.set_mode("train")
        self.agents.set_running_mode("train")
        if orig_learning_starts is not None:
            self.agents._learning_starts = orig_learning_starts
        if orig_gradient_steps is not None:
            self.agents._gradient_steps = orig_gradient_steps

        if not frames:
            return

        status_label = "success" if success else "fail"
        video_path = (
            video_dir
            / f"{self.rollout_video_prefix}_ep{episode_index:06d}_steps{steps}_{status_label}.mp4"
        )

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.rollout_video_fps,
            (width, height),
        )
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            writer.write(frame)
        writer.release()

        print(f"Saved rollout {episode_index} to {video_path}...")

        if rollout_log["states"]:
            json_path = video_path.with_suffix(".json")
            with json_path.open("w") as f:
                json.dump(rollout_log, f, indent=4)

            if self.log_rollout_path:
                path = Path(self.log_rollout_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w") as f:
                    json.dump(rollout_log, f, indent=4)
                self._rollout_log_saved = True
                if self.log_rollout_exit:
                    print(f"Saved rollout log to {path}. Exiting.")
                    sys.exit(0)

    def multi_agent_train(self) -> None:
        assert self.num_simultaneous_agents == 1
        assert self.env.num_agents > 1

        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
            range(self.initial_timestep, self.timesteps),
            disable=self.disable_progressbar,
            file=sys.stdout,
        ):
            self.agents.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                actions = self.agents.act(
                    states, timestep=timestep, timesteps=self.timesteps
                )[0]

                next_states, rewards, terminated, truncated, infos = self.env.step(
                    actions
                )
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                if not self.headless:
                    self.env.render()

                self.agents.record_transition(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                    infos=infos,
                    timestep=timestep,
                    timesteps=self.timesteps,
                )

                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            self.agents.post_interaction(
                terminated=terminated, timestep=timestep, timesteps=self.timesteps
            )

            if self.env.num_envs > 1:
                states = next_states
                shared_states = shared_next_states
            else:
                if terminated.any() or truncated.any():
                    with torch.no_grad():
                        states, infos = self.env.reset()
                        shared_states = self.env.state()
                else:
                    states = next_states
                    shared_states = shared_next_states

        self._maybe_save_rollout_log()

    def _maybe_log_rollout(
        self, states: torch.Tensor, actions: torch.Tensor, timestep: int
    ) -> None:
        if self.rollout_video_every_episodes:
            return
        if not self.log_rollout_path or self.log_rollout_steps <= 0:
            return
        if timestep >= self.log_rollout_steps or self._rollout_log_saved:
            return

        state_to_log = states
        if hasattr(self.agents, "_prev_states") and hasattr(self.agents, "_states"):
            if self.agents._prev_states is not None and self.agents._states is not None:
                state_to_log = torch.stack(
                    [self.agents._prev_states, self.agents._states], dim=1
                )

        self._rollout_log["states"].append(state_to_log.detach().cpu().tolist())
        self._rollout_log["actions"].append(actions.detach().cpu().tolist())

        if timestep + 1 >= self.log_rollout_steps:
            self._maybe_save_rollout_log()

    def _maybe_save_rollout_log(self) -> None:
        if self._rollout_log_saved or not self.log_rollout_path:
            return
        if not self._rollout_log["states"] and not self._rollout_log["actions"]:
            return
        path = Path(self.log_rollout_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._rollout_log, f, indent=4)
        self._rollout_log_saved = True
        if self.log_rollout_exit:
            print(f"Saved rollout log to {path}. Exiting.")
            sys.exit(0)
