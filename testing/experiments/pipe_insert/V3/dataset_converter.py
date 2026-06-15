from __future__ import annotations

import json
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def se3_z_axis_observation(
    R_current: np.ndarray,
    t_current: np.ndarray,
    R_target: np.ndarray,
    t_target: np.ndarray,
    include_z_dot: bool = True,
) -> np.ndarray:
    """Convert world-frame poses to [position error, relative target z-axis]."""
    dp_world = t_target - t_current
    dp = R_current.T @ dp_world
    z_rel = R_current.T @ R_target[:, 2]

    if include_z_dot:
        return np.concatenate([dp, z_rel], axis=0)
    return np.concatenate([dp, z_rel[:2]], axis=0)


def _relative_transform_to_z_axis_obs(
    transform: np.ndarray, include_z_dot: bool = True
) -> np.ndarray:
    if transform.shape != (4, 4):
        raise ValueError(f"Expected state with shape (4, 4), got {transform.shape}")

    # Full-SE3 demos store T_keypoint_target, so the requested observation is
    # the relative translation plus the target z-axis expressed in keypoint frame.
    dp = transform[:3, 3]
    z_rel = transform[:3, 2]

    if include_z_dot:
        return np.concatenate([dp, z_rel], axis=0).astype(np.float32)
    return np.concatenate([dp, z_rel[:2]], axis=0).astype(np.float32)


def _convert_sequence(
    states: list[Any], include_z_dot: bool = True
) -> list[list[float]]:
    converted = []
    for state in states:
        transform = np.asarray(state, dtype=np.float64)
        converted.append(
            _relative_transform_to_z_axis_obs(
                transform, include_z_dot=include_z_dot
            ).tolist()
        )
    return converted


def _convert_episode(data: dict[str, Any], include_z_dot: bool = True) -> dict[str, Any]:
    result = dict(data)

    states = _convert_sequence(data["states"], include_z_dot=include_z_dot)
    next_states = _convert_sequence(data["next_states"], include_z_dot=include_z_dot)

    result["states"] = states
    result["next_states"] = next_states
    if "observations" in result:
        result["observations"] = states
    result["observation_type"] = "align_z_and_pos"
    result["include_z_dot"] = include_z_dot

    return result


@dataclass
class Args:
    input_dir: Path = Path(
        "testing/experiments/pipe_insert/V3/datasets/subsample_x4_full_se3"
    )
    output_dir: Path = Path(
        "testing/experiments/pipe_insert/V3/datasets/align_z_and_pos"
    )
    include_z_dot: bool = True
    overwrite: bool = False


def main(args: Args | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--input-dir", type=Path, default=Args.input_dir)
        parser.add_argument("--output-dir", type=Path, default=Args.output_dir)
        parser.add_argument(
            "--include-z-dot",
            action=argparse.BooleanOptionalAction,
            default=Args.include_z_dot,
        )
        parser.add_argument(
            "--overwrite",
            action=argparse.BooleanOptionalAction,
            default=Args.overwrite,
        )
        args = Args(**vars(parser.parse_args()))

    if not args.input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {args.input_dir.as_posix()}"
        )

    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: "
            f"{args.output_dir.as_posix()}. Use --overwrite true to replace files."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(args.input_dir.glob("*.json"))
    if not input_files:
        raise FileNotFoundError(f"No json files found in {args.input_dir.as_posix()}")

    converted = 0
    total_steps = 0
    for input_path in input_files:
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        converted_data = _convert_episode(data, include_z_dot=args.include_z_dot)
        total_steps += len(converted_data["states"])

        output_path = args.output_dir / input_path.name
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(converted_data, f, indent=4)
        converted += 1

    obs_size = 6 if args.include_z_dot else 5
    print(f"Converted {converted} files")
    print(f"Steps: {total_steps}")
    print(f"Observation size: {obs_size}")
    print(f"Saved dataset to {args.output_dir.as_posix()}")


if __name__ == "__main__":
    main()
