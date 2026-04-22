"""
Convert ACT/ALOHA format HDF5 to the robotic failure dataset format.

Source format (episode_N.hdf5 root attrs):
  language_instruction   scalar str
  policy_class           scalar str
  sim                    bool

Source format datasets:
  /action                             (T, 14)  joint actions, both arms
  /base_action                        (T, 2)   mobile base commands
  /episode_annotations  (group, 0 datasets — metadata in attrs)
    [attr] episode_id         str  e.g. "episode_0"
    [attr] failure_category   str
    [attr] failure_description str
    [attr] success            float  1.0=success, 0.0=failure
    [attr] would_retry        float
  /observations/effort                (T, 14)  joint torques
  /observations/images/cam_high       (T, H, W, 3) uint8 RGB
  /observations/images/cam_left_wrist (T, H, W, 3) uint8 RGB
  /observations/images/cam_right_wrist(T, H, W, 3) uint8 RGB
  /observations/qpos                  (T, 14)  joint positions, both arms
  /observations/qvel                  (T, 14)  joint velocities, both arms

Target format (robotic failure dataset):
  action_dict/cartesian_position           [zeros — FK not available]
  action_dict/cartesian_velocity           [zeros — FK not available]
  action_dict/gripper_position        (T,)     zeros (bimanual — no single-arm gripper)
  action_dict/gripper_velocity        (T,)     zeros (bimanual — no single-arm gripper)
  image_observations/exterior_image_1_left  scalar str → mp4 path
  image_observations/exterior_image_2_left  scalar str → mp4 path
  image_observations/wrist_image_left       scalar str → mp4 path
  episode_annotations/episode_id      scalar str
  episode_annotations/success         scalar float
  episode_annotations/failure_category     scalar str
  episode_annotations/failure_description  scalar str
  episode_annotations/would_retry     scalar float
  episode_annotations/lab_id          scalar str
  episode_annotations/operator_name   scalar str
  episode_annotations/policy_id       scalar str
  episode_annotations/robot_id        scalar str
  episode_annotations/control_freq    scalar str
  observation/cartesian_position           [zeros — FK not available]
  observation/gripper_position        (T,)     from qpos[:, 6]
  observation/joint_position          (T, 14)  from qpos
  [file attr] language_instruction    scalar str  (read from source root attr)

Usage:
    python convert_ar_aloha_data.py -s /path/to/episode_0.hdf5 -o /path/to/output
    python convert_ar_aloha_data.py -s /path/to/episode_0.hdf5 -o /path/to/output -e 000001
    python convert_ar_aloha_data.py -s /path/to/Amazon-Aloha-Test/ -o /path/to/output
    python convert_ar_aloha_data.py -s /path/to/Amazon-Aloha-Test/ -o /path/to/output --start-id 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import imageio
import numpy as np


DEFAULT_CONTROL_FREQ = "30"

SCHEMA_VERSION = "robotic_failure_upload_data_format_v1"
MAX_DIM = 1080  # must match validate.py MAX_IMAGE_SIZE


def _write_mp4(video_path: Path, frames: np.ndarray, fps: float) -> None:
    """Write RGB frames (T, H, W, 3) to an MP4 file using libx264."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected frames with shape (T,H,W,3), got {frames.shape}")
    if frames.shape[0] == 0:
        raise ValueError("Cannot write video with zero frames")
    with imageio.get_writer(
        str(video_path),
        format="FFMPEG",
        mode="I",
        fps=float(fps),
        codec="libx264",
    ) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame, dtype=np.uint8))


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


def _decode_attr(value) -> str:
    """Decode an HDF5 attr value to a plain Python str."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _parse_fps(control_freq: str, default_fps: float = 30.0) -> float:
    try:
        parsed = float(control_freq)
    except (TypeError, ValueError):
        return default_fps
    return parsed if parsed > 0 else default_fps


def convert(
    source_h5: str,
    output_dir: Path,
    episode_id: str = "000000",
    control_freq: str = DEFAULT_CONTROL_FREQ,
    lab_id: str = "",
    operator_name: str = "",
    policy_id: str = "",
    robot_id: str = "",
) -> Path:
    """
    Convert a single ACT/ALOHA HDF5 file to the robotic failure dataset format.

    language_instruction, success, failure_category, failure_description, and
    would_retry are all read directly from the source file.

    Returns:
        Path to the output HDF5 file.
    """
    episode_id = str(episode_id).zfill(6)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_h5_path = output_dir / f"{episode_id}_trajectory.h5"
    str_dtype = h5py.string_dtype(encoding="utf-8")
    video_fps = _parse_fps(control_freq)

    print(f"\nReading source: {source_h5}")
    with h5py.File(source_h5, "r") as src:
        # Root attrs
        language_instruction = _decode_attr(src.attrs["language_instruction"])

        # Episode annotations (stored as group attrs in the new format)
        ea_attrs = src["episode_annotations"].attrs
        success = float(ea_attrs["success"])
        failure_category = _decode_attr(ea_attrs["failure_category"])
        failure_description = _decode_attr(ea_attrs["failure_description"])
        would_retry = float(ea_attrs["would_retry"])

        # Trajectory data
        action = src["action"][()]  # (T, 14)
        qpos = src["observations/qpos"][()]  # (T, 14)
        cam_high = src["observations/images/cam_high"][()]  # (T, H, W, 3)
        cam_lw = src["observations/images/cam_left_wrist"][()]  # (T, H, W, 3)
        cam_rw = src["observations/images/cam_right_wrist"][()]  # (T, H, W, 3)

    T = action.shape[0]
    print(f"Trajectory length: {T} steps")
    print(f"Language instruction: {language_instruction!r}")
    print(f"Success: {success}  |  would_retry: {would_retry}")
    print(f"Failure category: {failure_category!r}")

    # ── Video encoding ──────────────────────────────────────────────────────────
    # cam_high        → exterior_image_1_left
    # cam_right_wrist → exterior_image_2_left
    # cam_left_wrist  → wrist_image_left
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    ext1_rel = f"videos/{episode_id}_cam_high.mp4"
    ext2_rel = f"videos/{episode_id}_cam_right_wrist.mp4"
    wrist_rel = f"videos/{episode_id}_cam_left_wrist.mp4"

    print("\nEncoding videos (this may take a few minutes)...")
    _write_mp4(output_dir / ext1_rel, _resize_frames(cam_high), video_fps)
    _write_mp4(output_dir / ext2_rel, _resize_frames(cam_rw), video_fps)
    _write_mp4(output_dir / wrist_rel, _resize_frames(cam_lw), video_fps)

    # ── Observation fields ──────────────────────────────────────────────────────
    gripper_pos_obs = qpos[:, 6].astype(np.float32)

    # ── Write output HDF5 ───────────────────────────────────────────────────────
    print(f"\nWriting: {output_h5_path}")
    with h5py.File(output_h5_path, "w") as dst:
        dst.attrs["schema"] = SCHEMA_VERSION

        ea = dst.create_group("episode_annotations")
        ea.attrs["episode_id"] = episode_id
        ea.attrs["success"] = np.float32(success)
        ea.attrs["failure_category"] = failure_category
        ea.attrs["failure_description"] = failure_description
        ea.attrs["would_retry"] = np.float32(would_retry)
        ea.attrs["lab_id"] = lab_id
        ea.attrs["operator_name"] = operator_name
        ea.attrs["policy_id"] = policy_id
        ea.attrs["robot_id"] = robot_id
        ea.attrs["control_freq"] = control_freq

        img = dst.create_group("image_observations")
        img.create_dataset("exterior_image_1_left", data=ext1_rel, dtype=str_dtype)
        img.create_dataset("exterior_image_2_left", data=ext2_rel, dtype=str_dtype)
        img.create_dataset("wrist_image_left", data=wrist_rel, dtype=str_dtype)

        ad = dst.create_group("action_dict")
        ad.create_dataset(
            "cartesian_position", data=np.zeros((T, 6), dtype=np.float32)
        )
        ad.create_dataset(
            "cartesian_velocity", data=np.zeros((T, 6), dtype=np.float32)
        )
        ad.create_dataset("joint_position", data=action.astype(np.float32))
        ad.create_dataset("gripper_position", data=np.zeros(T, dtype=np.float32))
        ad.create_dataset("gripper_velocity", data=np.zeros(T, dtype=np.float32))

        obs = dst.create_group("observation")
        obs.create_dataset(
            "cartesian_position", data=np.zeros((T, 6), dtype=np.float32)
        )
        obs.create_dataset("gripper_position", data=gripper_pos_obs)
        obs.create_dataset("joint_position", data=qpos.astype(np.float32))

        dst.attrs["language_instruction"] = language_instruction

    print(f"\nDone. Wrote {output_h5_path}")
    return output_h5_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert ACT/ALOHA HDF5 to robotic failure dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source",
        "-s",
        required=True,
        help=(
            "Path to a single source HDF5 file (e.g. episode_0.hdf5) "
            "or a directory of HDF5 files to convert in bulk."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        type=Path,
        help="Output directory for converted files",
    )
    parser.add_argument(
        "--episode-id",
        "-e",
        default=None,
        help=(
            "Output episode ID (zero-padded to 6 digits, e.g. 000001). "
            "Only valid when --source is a single file. "
            "When converting a directory, IDs are assigned automatically starting from 0."
        ),
    )
    parser.add_argument(
        "--start-id",
        type=int,
        default=0,
        help="Starting episode ID when converting a directory (default: 0).",
    )
    parser.add_argument(
        "--control-freq",
        type=str,
        default=DEFAULT_CONTROL_FREQ,
        help=f"Control frequency in Hz, used as video FPS (default: {DEFAULT_CONTROL_FREQ})",
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
    args = parser.parse_args()

    source = Path(args.source)
    shared_kwargs = dict(
        output_dir=args.output_dir,
        control_freq=args.control_freq,
        lab_id=args.lab_id,
        operator_name=args.operator_name,
        policy_id=args.policy_id,
        robot_id=args.robot_id,
    )

    if source.is_dir():
        if args.episode_id is not None:
            parser.error(
                "--episode-id cannot be used with a directory source; use --start-id instead."
            )
        hdf5_files = sorted(source.glob("*.hdf5")) + sorted(source.glob("*.h5"))
        if not hdf5_files:
            parser.error(f"No .hdf5 or .h5 files found in {source}")
        print(f"\nFound {len(hdf5_files)} file(s) in {source}")
        failures = 0
        for i, h5_file in enumerate(hdf5_files):
            episode_id = str(args.start_id + i).zfill(6)
            print(f"\n{'=' * 60}")
            print(f"[{i + 1}/{len(hdf5_files)}] {h5_file.name} → episode {episode_id}")
            print(f"{'=' * 60}")
            try:
                convert(source_h5=str(h5_file), episode_id=episode_id, **shared_kwargs)
            except Exception as e:
                print(f"\n✗ Failed: {e}")
                failures += 1
        passed = len(hdf5_files) - failures
        print(f"\nDone: {passed}/{len(hdf5_files)} converted successfully.")
    else:
        episode_id = args.episode_id if args.episode_id is not None else "000000"
        convert(source_h5=str(source), episode_id=episode_id, **shared_kwargs)


if __name__ == "__main__":
    main()
