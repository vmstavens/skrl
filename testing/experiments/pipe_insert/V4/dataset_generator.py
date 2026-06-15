from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def _quat_normalize(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(q)
    if n < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def _rotvec_to_quat(v: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    angle = np.linalg.norm(v)
    if angle < eps:
        return _quat_normalize(
            np.array([1.0, 0.5 * v[0], 0.5 * v[1], 0.5 * v[2]], dtype=np.float64)
        )
    axis = v / angle
    half = 0.5 * angle
    sin_half = np.sin(half)
    return _quat_normalize(
        np.array(
            [np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
            dtype=np.float64,
        )
    )


def _quat_to_rotvec(q: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    q = _quat_normalize(q)
    if q[0] < 0:
        q = -q

    w = np.clip(q[0], -1.0, 1.0)
    xyz = q[1:]
    sin_half = np.linalg.norm(xyz)

    if sin_half < eps:
        return 2.0 * xyz

    axis = xyz / sin_half
    angle = 2.0 * np.arctan2(sin_half, w)
    return axis * angle


def _summarize_actions(actions: np.ndarray) -> np.ndarray:
    if actions.ndim != 2 or actions.shape[1] < 6:
        raise ValueError(f"Expected actions with shape [T, 6+], got {actions.shape}")

    pos_delta = actions[:, :3].sum(axis=0, dtype=np.float64)

    quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for rot_delta in actions[:, 3:6]:
        quat = _quat_mul(_rotvec_to_quat(rot_delta), quat)
    rot_delta = _quat_to_rotvec(quat)

    summarized = np.concatenate([pos_delta, rot_delta], axis=0).astype(np.float32)
    if actions.shape[1] > 6:
        summarized = np.concatenate(
            [
                summarized,
                actions[:, 6:].sum(axis=0, dtype=np.float64).astype(np.float32),
            ],
            axis=0,
        )
    return summarized


def _first_true_index(values: Iterable[bool]) -> int | None:
    for i, value in enumerate(values):
        if value:
            return i
    return None


def _subsample_episode(data: dict[str, Any], stride: int) -> dict[str, Any]:
    states = np.asarray(data["states"], dtype=np.float32)
    actions = np.asarray(data["actions"], dtype=np.float32)
    next_states = np.asarray(data["next_states"], dtype=np.float32)
    rewards = np.asarray(data["rewards"], dtype=np.float32)
    terminated = np.asarray(data["terminated"], dtype=bool)

    times = data.get("time")
    if times is not None:
        times = np.asarray(times, dtype=np.float64)

    n = min(len(states), len(actions), len(next_states), len(rewards), len(terminated))
    if n == 0:
        raise ValueError("Episode is empty")

    out_states = []
    out_actions = []
    out_next_states = []
    out_rewards = []
    out_terminated = []
    out_times = []

    start = 0
    while start < n:
        end = min(start + stride, n)
        term_offset = _first_true_index(terminated[start:end])
        if term_offset is not None:
            end = start + term_offset + 1

        out_states.append(states[start].tolist())
        out_actions.append(_summarize_actions(actions[start:end]).tolist())
        out_next_states.append(next_states[end - 1].tolist())
        out_rewards.append(float(rewards[start]))
        # out_rewards.append(float(rewards[start:end].sum(dtype=np.float64)))
        out_terminated.append(int(terminated[end - 1]))
        if times is not None:
            out_times.append(float(times[end - 1]))

        if terminated[end - 1]:
            break
        start = end

    effective_timestep = float(data.get("timestep", 0.0)) * stride
    result = {
        "timestamp": data.get("timestamp"),
        "timestep": effective_timestep,
        "states": out_states,
        "actions": out_actions,
        "next_states": out_next_states,
        "rewards": out_rewards,
        "terminated": out_terminated,
        "subsample_stride": stride,
    }
    if times is not None:
        result["time"] = out_times
    return result


@dataclass
class Args:
    input_dir: Path = Path("testing/experiments/pipe_insert/V4/demos/reduced_se3")
    output_dir: Path = Path("testing/experiments/pipe_insert/V4/datasets")
    stride: int = 3
    suffix: str = ""
    overwrite: bool = False


def main(args: Args | None = None) -> None:
    if args is None:
        import tyro

        args = tyro.cli(Args)

    if args.stride < 1:
        raise ValueError(f"stride must be >= 1, got {args.stride}")

    input_dir = args.input_dir
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir.as_posix()}")

    dataset_name = f"subsample_x{args.stride}_reduced_se3"
    if args.suffix:
        dataset_name = f"{dataset_name}_{args.suffix}"
    output_dir = args.output_dir / dataset_name

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir.as_posix()}. "
            "Use --overwrite true or change --suffix."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.json"))
    if not input_files:
        raise FileNotFoundError(f"No json demos found in {input_dir.as_posix()}")

    converted = 0
    total_input_steps = 0
    total_output_steps = 0

    for input_path in input_files:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        total_input_steps += len(data.get("actions", []))
        subsampled = _subsample_episode(data, stride=args.stride)
        total_output_steps += len(subsampled["actions"])

        output_path = output_dir / input_path.name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(subsampled, f, indent=4)
        converted += 1

    ratio = total_output_steps / max(total_input_steps, 1)
    print(f"Converted {converted} files")
    print(f"Input steps:  {total_input_steps}")
    print(f"Output steps: {total_output_steps}")
    print(f"Compression:  {ratio:.4f}")
    print(f"Saved dataset to {output_dir.as_posix()}")


if __name__ == "__main__":
    main()
