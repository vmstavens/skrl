import argparse
import json
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import glfw
import mujoco as mj
import numpy as np
import torch

from testing.base_sim import BaseSim, SimSync
from testing.mj import ObjType, get_pose
from testing.shen.diffusion_policy_state import (
    DIFFUSION_POLICY_STATE_DEFAULT_CONFIG,
    DiffusionPolicy,
)
from testing.shen.dp_models import ConditionalUnet1D, EMAModel


def mjs_cable(
    model_name: str = "cable",
    prefix: str = "cable:",
    curve: str = "0 s 0",
    count: str = "10 1 1",
    twist: float = 60000.0,
    bend: float = 10000000.0,
    vmax: float = 0,
    size: str = 1,
    segment_size: float = 0.002,
    mass: float = 0.00035,
    rgba: list = "0.2 0.2 0.2 1",
    initial: str = "free",
) -> mj.MjSpec:
    _XML = f"""
<mujoco model="{model_name}">
    <extension>
        <plugin plugin="mujoco.elasticity.cable"/>
    </extension>

    <worldbody>
        <composite prefix="{prefix}" type="cable" curve="{curve}" count="{count}" size="{size}" initial="{initial}">
            <plugin plugin="mujoco.elasticity.cable">
                <config key="twist" value="{twist}" />
                <config key="bend" value="{bend}" />
                <config key="vmax" value="{vmax}" />
            </plugin>
            <joint kind="main" damping="1e-2" armature="0.001" />
            <geom type="capsule"
                size="{segment_size}"
                rgba="{rgba}"
                mass="{mass}"
                friction="0.3 0.3 0.3"
                condim="4"
                solref="0.001 3"
            />
        </composite>
    </worldbody>

</mujoco>
    """

    return mj.MjSpec.from_string(_XML)


def pipe2(length: float = 0.1) -> mj.MjSpec:
    _XML = f"""
    <mujoco>
        <worldbody>
            <body euler="0 0 0" pos="0 0 0">
                <replicate sep="hole:" count="30" euler="0 0 20">
                    <geom type="box" solref="0.0000000001 1" pos="0 -0.018 0" size=".004 .001 {length / 2}" friction="0.2 0.2 0.2" />
                </replicate>
            </body>
        </worldbody>
    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


def empty() -> mj.MjSpec:
    _XML = """
        <mujoco model="empty scene">

        <compiler angle="radian" autolimits="true" />
        <option timestep="0.002"
            integrator="implicitfast"
            solver="Newton"
            gravity="0 0 -9.82"
            cone="elliptic"
            sdf_iterations="5"
            sdf_initpoints="30"
            noslip_iterations="2"
            ls_iterations="10"
        >
                <flag eulerdamp="disable" />
        </option>

        <custom>
            <numeric data="15" name="max_contact_points" />
            <numeric data="15" name="max_geom_pairs" />
        </custom>

        <extension>
            <plugin plugin="mujoco.sensor.touch_grid" />
        </extension>

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
        </worldbody>

    </mujoco>
    """
    return mj.MjSpec().from_string(_XML)


def build_pipe_insert_model(slide_limits: dict[str, tuple[float, float]]) -> mj.MjModel:
    scene = empty()
    pip = pipe2()

    pip.worldbody.first_body().add_site(
        name="target_1", pos=[0, 0.015, -0.05], group=1, rgba=[1, 0, 0, 1]
    )
    pip.worldbody.first_body().add_site(
        name="target_2", pos=[0, 0.015, 0], group=1, rgba=[1, 1, 0, 1]
    )
    pip.worldbody.first_body().add_site(
        name="target_3", pos=[0, 0.015, 0.05], group=1, rgba=[0, 1, 0, 1]
    )

    cable = mjs_cable(count="40 1 1", twist=60000.0 * 2, bend=10000000.0 * 2)
    cable.bodies[1].add_site(name="keypoint_3", group=1, rgba=[0, 1, 0, 1])
    cable.bodies[3].add_site(name="keypoint_2", group=1, rgba=[1, 1, 0, 1])
    cable.bodies[5].add_site(name="keypoint_1", group=1, rgba=[1, 0, 0, 1])

    scene.worldbody.add_camera(
        name="cam",
        pos=[0.721, 0.234, 0.156],
        xyaxes=[-0.037, 0.999, 0.000, -0.001, -0.000, 1.000],
    )

    gripper = scene.worldbody.add_body(
        name="gripper", pos=[0, 0.4, 0.1], euler=[0, 0, 1.57]
    )
    gripper.add_geom(
        type=mj.mjtGeom.mjGEOM_BOX,
        size=[0.02, 0.02, 0.02],
        contype=0,
        conaffinity=0,
    )
    gripper.add_joint(
        name="x",
        type=mj.mjtJoint.mjJNT_SLIDE,
        axis=[1, 0, 0],
        range=slide_limits["x"],
    )
    gripper.add_joint(
        name="z",
        type=mj.mjtJoint.mjJNT_SLIDE,
        axis=[0, 0, 1],
        range=slide_limits["z"],
    )
    scene.add_actuator(
        name="x",
        target="x",
        trntype=mj.mjtTrn.mjTRN_JOINT,
        ctrlrange=slide_limits["x"],
    ).set_to_position(kp=100, kv=100)
    scene.add_actuator(
        name="z",
        target="z",
        trntype=mj.mjtTrn.mjTRN_JOINT,
        ctrlrange=slide_limits["z"],
    ).set_to_position(kp=100, kv=100)

    scene.worldbody.add_frame(pos=[0, 0, 0.1], euler=[1.57, 0, 3.14]).attach_body(
        pip.worldbody.first_body()
    )
    scene.worldbody.add_frame(pos=[0, 0.2, 0.1], euler=[0, 0, 0]).attach_body(
        cable.worldbody.first_body()
    )

    b = None
    b0 = cable.worldbody.first_body()
    for _ in range(10):
        b = b0.first_body()
        b0 = b

    scene.add_equality(
        name="weld",
        type=mj.mjtEq.mjEQ_WELD,
        objtype=mj.mjtObj.mjOBJ_BODY,
        name1="gripper",
        name2=b.name,
        data=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        solref=[0.000001, 2],
    )

    return scene.compile()


def _ensure_stats_arrays(stats: dict) -> dict:
    for key in ("obs", "action"):
        stats[key]["min"] = np.asarray(stats[key]["min"], dtype=np.float32)
        stats[key]["max"] = np.asarray(stats[key]["max"], dtype=np.float32)
    return stats


def load_stats(stats_path: Path) -> dict:
    if stats_path.suffix == ".pkl":
        import pickle

        with stats_path.open("rb") as f:
            stats = pickle.load(f)
    else:
        with stats_path.open("r") as f:
            stats = json.load(f)

    return _ensure_stats_arrays(stats)


class PipeInsertSim(BaseSim):
    def __init__(
        self,
        checkpoint_path: Path,
        stats_path: Path,
        device: Optional[str] = None,
    ):
        super().__init__()
        self._slide_limits = {"x": (-0.3, 0.3), "z": (-0.3, 0.3)}

        self._model = build_pipe_insert_model(self._slide_limits)
        self._data = mj.MjData(self._model)

        self._stats = load_stats(stats_path)
        self._policy, loaded_stats = self._build_policy(
            checkpoint_path, device=device, stats=self._stats
        )
        if loaded_stats is not None:
            self._stats = loaded_stats

        self._obs_horizon = self._policy.config["obs_horizon"]
        self._action_horizon = self._policy.config["action_horizon"]
        self._obs_deque: Deque[np.ndarray] = deque(maxlen=self._obs_horizon)
        self._pending_actions: Deque[np.ndarray] = deque()
        self._prev_site_pos: Optional[np.ndarray] = None

        self._target_ids = [self._model.site(f"target_{i + 1}").id for i in range(3)]
        self._keypoint_ids = [
            self._model.site(f"keypoint_{i + 1}").id for i in range(3)
        ]

        self._ctrl_low = self._model.actuator_ctrlrange[:, 0]
        self._ctrl_high = self._model.actuator_ctrlrange[:, 1]

        self._initial_qpos = np.load(
            "testing/experiments/pipe_insert/constants/qpos.npy"
        )
        self._initial_qvel = np.load(
            "testing/experiments/pipe_insert/constants/qvel.npy"
        )
        self._initial_ctrl = np.load(
            "testing/experiments/pipe_insert/constants/ctrl.npy"
        )

        self.reset_state()
        self.threads = [self.policy_loop]

    @property
    def data(self) -> mj.MjData:
        return self._data

    @property
    def model(self) -> mj.MjModel:
        return self._model

    def reset_state(self) -> None:
        self._data.qpos[:] = self._initial_qpos
        self._data.qvel[:] = self._initial_qvel
        self._data.ctrl[:] = self._initial_ctrl
        mj.mj_forward(self._model, self._data)
        self._obs_deque.clear()
        self._pending_actions.clear()
        self._prev_site_pos = None

    def _build_policy(
        self, checkpoint_path: Path, device: Optional[str], stats: Optional[dict]
    ) -> Tuple[DiffusionPolicy, Optional[dict]]:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path.as_posix(), map_location=device)
        dp_config = DIFFUSION_POLICY_STATE_DEFAULT_CONFIG.copy()
        if "config" in checkpoint and isinstance(checkpoint["config"], dict):
            dp_config.update(checkpoint["config"])

        checkpoint_stats = checkpoint.get("stats")
        if checkpoint_stats is not None:
            stats = _ensure_stats_arrays(checkpoint_stats)

        if stats is None:
            raise ValueError("Stats are required from the checkpoint or --stats.")

        o_dim = int(checkpoint.get("o_dim", stats["obs"]["min"].shape[0]))
        a_dim = int(checkpoint.get("a_dim", stats["action"]["min"].shape[0]))
        dp_config["obs_dim"] = o_dim
        dp_config["global_cond_dim"] = dp_config["obs_horizon"] * dp_config["obs_dim"]

        dp_models = {}
        dp_models["model"] = ConditionalUnet1D(
            a_dim=a_dim, o_dim=o_dim, config=dp_config
        ).to(device)
        ema = EMAModel(dp_models["model"].parameters(), power=dp_config["ema_power"])
        dp_models["ema_model"] = ConditionalUnet1D(
            a_dim=a_dim, o_dim=o_dim, config=dp_config
        ).to(device)

        policy = DiffusionPolicy(
            a_dim=a_dim,
            o_dim=o_dim,
            models=dp_models,
            ema=ema,
            device=device,
            config=dp_config,
        )

        if "model_state_dict" in checkpoint:
            policy.model.load_state_dict(checkpoint["model_state_dict"])
            policy.ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
        elif "policy" in checkpoint:
            policy.model.load_state_dict(checkpoint["policy"])
            if "ema_model" in checkpoint:
                policy.ema_model.load_state_dict(checkpoint["ema_model"])
        else:
            raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")

        policy.set_mode("eval")
        policy.ema.copy_to(policy.ema_model.parameters())
        policy.stats = stats
        return policy, stats

    def _get_site_pos(self) -> np.ndarray:
        target_positions = []
        key_positions = []
        for tid, kid in zip(self._target_ids, self._keypoint_ids):
            T_w_target = get_pose(self._model, self._data, tid, ObjType.SITE)
            T_w_key = get_pose(self._model, self._data, kid, ObjType.SITE)
            target_positions.append(T_w_target.t)
            key_positions.append(T_w_key.t)

        dist = [
            np.linalg.norm(target_positions[i] - key_positions[i]) for i in range(3)
        ]
        return np.array(dist, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        s1 = self._get_site_pos()
        if self._prev_site_pos is None:
            self._prev_site_pos = s1
        dt = float(self._model.opt.timestep)
        v1 = (s1 - self._prev_site_pos) / dt
        self._prev_site_pos = s1
        return np.concatenate([s1, v1]).astype(np.float32)

    def _infer_actions(self) -> list[np.ndarray]:
        obs_seq = np.stack(self._obs_deque)
        obs_tensor = (
            torch.from_numpy(obs_seq)
            .to(self._policy.device, dtype=torch.float32)
            .unsqueeze(0)
        )

        with torch.no_grad():
            actions_pred, _, _ = self._policy.act(states=obs_tensor)

        start = self._obs_horizon - 1
        end = start + self._action_horizon
        actions = actions_pred[0, start:end, :].detach().cpu().numpy()
        return [np.asarray(a, dtype=np.float32) for a in actions]

    def _apply_action(self, action: np.ndarray) -> None:
        ctrl = np.clip(self._data.ctrl + action, self._ctrl_low, self._ctrl_high)
        self._data.ctrl[:] = ctrl

    def policy_loop(self, ss: SimSync) -> None:
        while True:
            ss.step()
            obs = self._get_obs()
            if not self._obs_deque:
                for _ in range(self._obs_horizon):
                    self._obs_deque.append(obs)
            else:
                self._obs_deque.append(obs)

            if not self._pending_actions:
                self._pending_actions.extend(self._infer_actions())

            if self._pending_actions:
                action = self._pending_actions.popleft()
                self._apply_action(action)

    def keyboard_callback(self, key: int) -> None:
        if key is glfw.KEY_SPACE:
            self.reset_state()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("testing/experiments/pipe_insert/.runs/data_100/models/latest_model.pth"),
        help="Path to diffusion policy checkpoint.",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("testing/experiments/pipe_insert/.stats/pipe_insert_stats.json"),
        help="Path to normalization stats JSON/PKL.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g., cpu or cuda).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sim = PipeInsertSim(
        checkpoint_path=args.checkpoint,
        stats_path=args.stats,
        device=args.device,
    )
    sim.run()
