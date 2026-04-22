"""
Convert RLDS dataset to HDF5 format. This script assumes that the dataset is stored in the RDLS DROID format.

If your dataset field names differ, please modify the CONVERSION_DICT below to map from expected output field names to your dataset's field names. The expected output format is as follows:
- Each episode is stored as a separate HDF5 file containing:
  - An "observation" group with datasets for "gripper_position", "cartesian_position", and "joint_position".
  - An "action_dict" group with datasets for "gripper_position", "gripper_velocity", "cartesian_position", and "cartesian_velocity".
  - A "language_instruction" dataset containing the chosen language instruction text.
  - Optionally, an "episode_annotations" group containing metadata fields such as "lab_id", "operator_name", "policy_id", "robot_id", "control_freq", and "success".
- Image observations are stored as separate MP4 video files, and the HDF5 contains a dataset with the relative path to each video file. The script looks for image observations in the "observation" dict and expects them to be 4D uint8 arrays with shape [T, H, W, 3]. If found, it writes them as MP4 videos and stores the relative path in the HDF5 under the "image_observations" group. The video filename is derived from the episode filename and the observation key, sanitized to be filesystem-safe. The script uses a default frame rate of 15 FPS for the videos, which can be overridden with the --control-freq argument if your dataset includes a control frequency annotation.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np

from oopsie_tools.annotation_tool.episode_recorder import write_mp4

SCHEMA_VERSION = "robotic_failure_upload_data_format_v1"
MAX_DIM = 1080  # must match validate.py MAX_IMAGE_SIZE


def _resize_frames(frames: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    """Resize (T, H, W, 3) frames so that max(H, W) <= max_dim."""
    h, w = frames.shape[1:3]
    if w <= max_dim and h <= max_dim:
        return frames
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return np.stack(
        [cv2.resize(f, (new_w, new_h), interpolation=cv2.INTER_AREA) for f in frames]
    )


def _write_episode_annotations(
    file_handle: h5py.File,
    episode_id: str,
    success: float,
    lab_id: str = "",
    operator_name: str = "",
    policy_id: str = "",
    robot_id: str = "",
    control_freq: str = "",
) -> None:
    """Create and populate the episode_annotations group in *file_handle*."""
    str_dtype = h5py.string_dtype(encoding="utf-8")
    group = file_handle.create_group("episode_annotations")
    group.create_dataset("episode_id", data=episode_id, dtype=str_dtype)
    if lab_id.strip():
        group.create_dataset("lab_id", data=lab_id, dtype=str_dtype)
    if operator_name.strip():
        group.create_dataset("operator_name", data=operator_name, dtype=str_dtype)
    if policy_id.strip():
        group.create_dataset("policy_id", data=policy_id, dtype=str_dtype)
    if robot_id.strip():
        group.create_dataset("robot_id", data=robot_id, dtype=str_dtype)
    if control_freq.strip():
        group.create_dataset("control_freq", data=control_freq, dtype=str_dtype)
    group.create_dataset("success", data=np.float32(success))


CONVERSION_DICT = {
    # Observation fields
    "observation/gripper_position": "observation/gripper_position",
    "observation/cartesian_position": "observation/cartesian_position",
    "observation/joint_position": "observation/joint_position",
    # Action fields
    "action_dict/gripper_position": "action_dict/gripper_position",
    "action_dict/gripper_velocity": "action_dict/gripper_velocity",
    "action_dict/cartesian_position": "action_dict/cartesian_position",
    "action_dict/cartesian_velocity": "action_dict/cartesian_velocity",
    # Language instruction fields (we only select one language instruction per episode)
    "language_instruction": "steps/language_instruction",
}


def _stack_step_records(values: list[Any]) -> Any:
    if not values:
        return np.asarray([])

    first = values[0]
    if isinstance(first, dict):
        return {
            key: _stack_step_records([value[key] for value in values])
            for key in first.keys()
        }

    return np.asarray(values)


def _materialize_steps(episode: dict[str, Any]) -> dict[str, Any]:
    """Return RLDS steps as a dict-of-arrays for both RLDS storage styles.

    Some TFDS/RLDS pipelines expose episode["steps"] as a dict of arrays,
    while others expose it as an iterable dataset of per-step dict records.
    """
    steps = episode.get("steps")
    if isinstance(steps, dict):
        return steps

    if steps is None:
        raise KeyError("Episode is missing 'steps'")

    try:
        iterator = iter(steps)
    except TypeError as exc:
        raise TypeError(f"Unsupported 'steps' container type: {type(steps)!r}") from exc

    records = list(iterator)
    if not records:
        raise ValueError("Episode has no steps")

    if not isinstance(records[0], dict):
        raise TypeError(
            f"Expected iterable 'steps' records to be dicts, got {type(records[0])!r}"
        )

    return _stack_step_records(records)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_text(value.item())
        if value.size == 0:
            return ""
        first = value.reshape(-1)[0]
        return _decode_text(first)
    return str(value)


def _choose_language_instruction(steps: dict[str, Any]) -> str:
    for key in (
        "language_instruction",
        "language_instruction_2",
        "language_instruction_3",
    ):
        if key not in steps:
            continue
        values = np.asarray(steps[key])
        if values.size == 0:
            continue
        for item in values.reshape(-1):
            decoded = _decode_text(item).strip()
            if decoded:
                return decoded
    return ""


def _infer_success(steps: dict[str, Any]) -> float:
    rewards = np.asarray(steps.get("reward", []), dtype=np.float32)
    if rewards.size > 0:
        return float(rewards[-1])
    is_terminal = np.asarray(steps.get("is_terminal", []), dtype=bool)
    if is_terminal.size > 0:
        return float(is_terminal[-1])
    return 0.0


def _sanitize_stem(text: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    sanitized = "".join(ch if ch in allowed else "_" for ch in text)
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized.strip("_") or "episode"


def _episode_stem(episode_idx: int, episode_metadata: dict[str, Any]) -> str:
    source_path = _decode_text(episode_metadata.get("file_path", ""))
    if source_path:
        source_stem = _sanitize_stem(Path(source_path).stem)
        return f"{episode_idx:06d}_{source_stem}"
    return f"{episode_idx:06d}"


def _parse_fps(control_freq: str, default_fps: float = 15.0) -> float:
    try:
        parsed = float(control_freq)
    except (TypeError, ValueError):
        return default_fps
    if parsed <= 0:
        return default_fps
    return parsed


def _split_path(path: str) -> list[str]:
    return [part for part in path.split("/") if part]


def _get_nested_value(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in _split_path(path):
        if not isinstance(current, dict):
            raise KeyError(f"Path '{path}' is not valid at '{part}'.")
        current = current[part]
    return current


def _set_h5_dataset(
    file_handle: h5py.File, target_path: str, value: Any, dtype: Any
) -> None:
    parts = _split_path(target_path)
    if not parts:
        raise ValueError("Target path cannot be empty.")

    group: h5py.File | h5py.Group = file_handle
    for part in parts[:-1]:
        group = group.require_group(part)

    group.create_dataset(parts[-1], data=value, dtype=dtype)


def _write_mapped_datasets(
    file_handle: h5py.File, steps: dict[str, Any], str_dtype: Any
) -> None:
    source_root: dict[str, Any] = {"steps": steps, **steps}

    for target_path, source_path in CONVERSION_DICT.items():
        if target_path == "language_instruction":
            continue

        source_value = _get_nested_value(source_root, source_path)
        _set_h5_dataset(
            file_handle=file_handle,
            target_path=target_path,
            value=np.asarray(source_value, dtype=np.float64),
            dtype=np.float64,
        )

    language_source_path = CONVERSION_DICT.get("language_instruction")
    if language_source_path is None:
        language_instruction = _choose_language_instruction(steps)
    else:
        language_values = np.asarray(
            _get_nested_value(source_root, language_source_path)
        )
        language_instruction = ""
        if language_values.size > 0:
            for item in language_values.reshape(-1):
                decoded = _decode_text(item).strip()
                if decoded:
                    language_instruction = decoded
                    break
        if not language_instruction:
            language_instruction = _choose_language_instruction(steps)

    _set_h5_dataset(
        file_handle=file_handle,
        target_path="language_instruction",
        value=language_instruction,
        dtype=str_dtype,
    )


def _write_episode_h5(
    output_path: Path,
    episode_idx: int,
    episode: dict[str, Any],
    lab_id: str,
    operator_name: str,
    policy_id: str,
    robot_id: str,
    control_freq: str,
    compression_level: int,
    store_episode_annotations: bool,
) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    steps = _materialize_steps(episode)
    observation = steps["observation"]
    success = np.float32(_infer_success(steps))
    episode_id = _episode_stem(
        episode_idx=episode_idx, episode_metadata=episode.get("episode_metadata", {})
    )
    video_fps = _parse_fps(control_freq)

    with h5py.File(output_path, "w") as f:
        f.attrs["schema"] = SCHEMA_VERSION

        if store_episode_annotations:
            _write_episode_annotations(
                file_handle=f,
                episode_id=episode_id,
                success=float(success),
                lab_id=lab_id,
                operator_name=operator_name,
                policy_id=policy_id,
                robot_id=robot_id,
                control_freq=control_freq,
            )

        image_group = f.create_group("image_observations")
        for key, value in observation.items():
            arr = np.asarray(value)
            if arr.ndim == 4 and arr.shape[-1] == 3 and arr.dtype == np.uint8:
                safe_key = _sanitize_stem(key)
                video_path = output_path.parent / f"{output_path.stem}__{safe_key}.mp4"
                write_mp4(
                    video_path=video_path, frames=_resize_frames(arr), fps=video_fps
                )
                rel_video_path = os.path.relpath(
                    video_path.resolve(),
                    start=output_path.parent.resolve(),
                )
                image_group.create_dataset(
                    key, data=rel_video_path.replace(os.sep, "/"), dtype=str_dtype
                )

        _write_mapped_datasets(
            file_handle=f,
            steps=steps,
            str_dtype=str_dtype,
        )


def convert_rlds_to_hdf5(
    rlds_version_dir: Path,
    output_dir: Path,
    split: str,
    max_episodes: int | None,
    lab_id: str,
    operator_name: str,
    policy_id: str,
    robot_id: str,
    control_freq: str,
    compression_level: int,
    store_episode_annotations: bool,
    overwrite: bool,
) -> None:
    try:
        import tensorflow_datasets as tfds
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "tensorflow_datasets is required for RLDS conversion. "
            "Install TFDS deps with: uv sync --extra tfds"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(str(rlds_version_dir))
    dataset = builder.as_dataset(split=split)

    written = 0
    for idx, episode in enumerate(tfds.as_numpy(dataset)):
        if max_episodes is not None and written >= max_episodes:
            break

        stem = _episode_stem(
            episode_idx=idx, episode_metadata=episode.get("episode_metadata", {})
        )
        out_path = output_dir / f"{stem}.h5"
        if out_path.exists() and not overwrite:
            continue

        _write_episode_h5(
            output_path=out_path,
            episode_idx=idx,
            episode=episode,
            lab_id=lab_id,
            operator_name=operator_name,
            policy_id=policy_id,
            robot_id=robot_id,
            control_freq=control_freq,
            compression_level=compression_level,
            store_episode_annotations=store_episode_annotations,
        )
        written += 1
        if written % 20 == 0:
            print(f"Converted {written} episodes...")

    print(f"Done. Wrote {written} trajectory HDF5 files to: {output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert RLDS TFDS shards into trajectory-level HDF5 files."
    )
    parser.add_argument(
        "--rlds-version-dir",
        type=Path,
        default=Path("1.0.0"),
        help="Path to RLDS TFDS version directory (contains dataset_info.json and TFRecord shards).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("trajectory_hdf5"),
        help="Output directory for generated trajectory-level .h5 files.",
    )
    parser.add_argument(
        "--split", type=str, default="train", help="Dataset split to convert."
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional cap on number of episodes to convert.",
    )
    parser.add_argument(
        "--lab-id", type=str, default="", help="Value for episode_annotations/lab_id."
    )
    parser.add_argument(
        "--operator-name",
        type=str,
        default="",
        help="Value for episode_annotations/operator_name.",
    )
    parser.add_argument(
        "--policy-id",
        type=str,
        default="",
        help="Value for episode_annotations/policy_id.",
    )
    parser.add_argument(
        "--robot-id",
        type=str,
        default="",
        help="Value for episode_annotations/robot_id.",
    )
    parser.add_argument(
        "--control-freq",
        type=str,
        default="",
        help="Value for episode_annotations/control_freq.",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=4,
        help="gzip compression level for image datasets (0-9).",
    )
    parser.add_argument(
        "--no-store-episode-annotations",
        dest="store_episode_annotations",
        action="store_false",
        default=True,
        help="Skip storing episode_annotations group in output HDF5.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output files if they already exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not (0 <= args.compression_level <= 9):
        raise ValueError("--compression-level must be in [0, 9].")

    convert_rlds_to_hdf5(
        rlds_version_dir=args.rlds_version_dir,
        output_dir=args.output_dir,
        split=args.split,
        max_episodes=args.max_episodes,
        lab_id=args.lab_id,
        operator_name=args.operator_name,
        policy_id=args.policy_id,
        robot_id=args.robot_id,
        control_freq=args.control_freq,
        compression_level=args.compression_level,
        store_episode_annotations=args.store_episode_annotations,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
