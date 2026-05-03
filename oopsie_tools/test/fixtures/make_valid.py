#!/usr/bin/env python3
"""Generate a set of valid test HDF5 episodes with synthetic MP4 videos.

Writes files into OUTPUT_DIR (default: oopsie_tools/test/fixtures/samples/).
Each fixture covers a distinct annotation scenario so that the annotation tool,
annotator server, and any downstream readers can be exercised without real data.

Scenarios created
-----------------
episode_unannotated     – valid episode, no annotation group
episode_success         – annotated as success by "test_annotator"
episode_failure         – annotated as failure with taxonomy by "test_annotator"
episode_multi_camera    – two cameras (left + wrist), no annotation

Usage
-----
    uv run python -m oopsie_tools.test.fixtures.make_valid [OUTPUT_DIR]

The videos are solid-color 64×64 MP4s (10 frames each).  They are tiny
(< 20 KB) and require only imageio + libx264.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import imageio
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

_ROBOT_PROFILE = json.dumps(
    {
        "policy_name": "test_policy",
        "robot_name": "test_robot",
        "is_biarm": False,
        "uses_mobile_base": False,
        "gripper_name": "test_gripper",
        "control_freq": 10,
        "camera_names": ["front"],
        "robot_state_keys": ["joint_position", "gripper_position"],
        "robot_state_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        "action_space": ["joint_velocity", "gripper_position"],
        "action_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        "orientation_representation": None,
        "controller": None,
        "gains": None,
        "intrinsic_calibration_matrix": None,
        "extrinsic_calibration_matrix": None,
    }
)

_ROBOT_PROFILE_MULTI_CAM = json.dumps(
    {
        **json.loads(_ROBOT_PROFILE),
        "camera_names": ["left", "wrist"],
    }
)


def _write_video(path: Path, color: tuple[int, int, int], n_frames: int = 25, size: int = 224) -> None:
    frames = np.full((n_frames, size, size, 3), color, dtype=np.uint8)
    with imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=10,
        codec="libx264",
        output_params=["-crf", "28"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _robot_states(n: int = 20) -> dict[str, np.ndarray]:
    return {
        "joint_position": np.random.randn(n, 7).astype(np.float64),
        "gripper_position": np.random.randn(n, 1).astype(np.float64),
    }


def _actions(n: int = 20) -> dict[str, np.ndarray | None]:
    return {
        "joint_velocity": np.random.randn(n, 7).astype(np.float64),
        "gripper_position": np.random.randn(n, 1).astype(np.float64),
        "cartesian_position": None,
        "cartesian_velocity": None,
        "joint_position": None,
        "base_position": None,
        "base_velocity": None,
        "gripper_velocity": None,
        "gripper_binary": None,
    }


def _write_base_h5(
    f: h5py.File,
    *,
    episode_id: str,
    language_instruction: str,
    camera_video_paths: dict[str, str],
    robot_profile_json: str = _ROBOT_PROFILE,
    n_steps: int = 20,
) -> None:
    f.attrs["schema"] = "oopsiedata_format_v1"
    f.attrs["episode_id"] = episode_id
    f.attrs["robot_profile"] = robot_profile_json
    f.attrs["language_instruction"] = language_instruction
    f.attrs["operator_name"] = "test_operator"
    f.attrs["lab_id"] = "test_lab"
    f.attrs["timestamp"] = time.time()

    img_obs = f.create_group("image_observations")
    for cam, rel_path in camera_video_paths.items():
        img_obs.create_dataset(cam, data=rel_path, dtype=_STR_DTYPE)

    obs_group = f.create_group("observations")
    obs_group.create_group("video_paths")
    rs = obs_group.create_group("robot_states")
    for key, arr in _robot_states(n_steps).items():
        rs.create_dataset(key, data=arr, dtype=np.float64)

    actions_group = f.create_group("actions")
    for key, val in _actions(n_steps).items():
        if val is None:
            actions_group.create_dataset(key, data=h5py.Empty(dtype=np.float64))
        else:
            actions_group.create_dataset(key, data=val, dtype=np.float64)


def _write_annotation(
    f: h5py.File,
    *,
    annotator: str,
    success: float,
    failure_description: str,
    failure_category: list[str],
    severity: str,
    additional_notes: str = "",
) -> None:
    ea = f.require_group("episode_annotations")
    ag = ea.require_group(annotator)
    ag.attrs["schema"] = "oopsie_failure_taxonomy_v1"
    ag.attrs["source"] = "human"
    ag.attrs["timestamp"] = "2026-04-21T10:00:00+00:00"
    ag.attrs["success"] = success
    ag.attrs["failure_description"] = failure_description
    ag.attrs["taxonomy_schema"] = "oopsiedata_taxonomy_schema_v1"
    ag.attrs["taxonomy"] = json.dumps(
        {"failure_category": failure_category, "severity": severity},
        ensure_ascii=False,
    )
    ag.attrs["additional_notes"] = additional_notes


# ---------------------------------------------------------------------------
# Public helper for per-test fixture generation
# ---------------------------------------------------------------------------


def write_valid_episode(
    out_dir: Path,
    stem: str = "episode",
    *,
    cam_color: tuple[int, int, int] = (120, 160, 200),
    n: int = 20,
) -> Path:
    """Write a single valid episode (HDF5 + MP4) into *out_dir* and return the HDF5 path.

    Intended for use in pytest fixtures that need an isolated valid episode,
    e.g. ``tmp_path``-based per-test fixtures.
    """
    mp4_name = f"{stem}_front.mp4"
    _write_video(out_dir / mp4_name, color=cam_color)
    h5_path = out_dir / f"{stem}.h5"
    with h5py.File(h5_path, "w") as f:
        _write_base_h5(
            f,
            episode_id=stem,
            language_instruction="pick up the block",
            camera_video_paths={"front": mp4_name},
            n_steps=n,
        )
        _write_annotation(
            f,
            annotator="test_annotator",
            success=1.0,
            failure_description="",
            failure_category=[],
            severity="none",
        )
    return h5_path


# ---------------------------------------------------------------------------
# Fixture writers
# ---------------------------------------------------------------------------


def make_unannotated(out_dir: Path) -> None:
    _write_video(out_dir / "episode_unannotated_front.mp4", color=(80, 120, 200))
    with h5py.File(out_dir / "episode_unannotated.h5", "w") as f:
        _write_base_h5(
            f,
            episode_id="episode_unannotated",
            language_instruction="pick up the red block",
            camera_video_paths={"front": "episode_unannotated_front.mp4"},
        )
        _write_annotation(
            f,
            annotator="test_annotator",
            success=1.0,
            failure_description="",
            failure_category=[],
            severity="none",
        )


def make_success(out_dir: Path) -> None:
    _write_video(out_dir / "episode_success_front.mp4", color=(60, 180, 80))
    with h5py.File(out_dir / "episode_success.h5", "w") as f:
        _write_base_h5(
            f,
            episode_id="episode_success",
            language_instruction="place the cup on the tray",
            camera_video_paths={"front": "episode_success_front.mp4"},
        )
        _write_annotation(
            f,
            annotator="test_annotator",
            success=1.0,
            failure_description="",
            failure_category=[],
            severity="none",
            additional_notes="Clean success, no issues.",
        )


def make_failure(out_dir: Path) -> None:
    _write_video(out_dir / "episode_failure_front.mp4", color=(200, 60, 60))
    with h5py.File(out_dir / "episode_failure.h5", "w") as f:
        _write_base_h5(
            f,
            episode_id="episode_failure",
            language_instruction="stack the two blocks",
            camera_video_paths={"front": "episode_failure_front.mp4"},
        )
        _write_annotation(
            f,
            annotator="test_annotator",
            success=0.0,
            failure_description="Robot grasped the block but dropped it during transport.",
            failure_category=["grasp_failure", "transport_failure"],
            severity="major",
            additional_notes="Happens consistently at the same waypoint.",
        )


def make_multi_camera(out_dir: Path) -> None:
    _write_video(out_dir / "episode_multi_camera_left.mp4", color=(160, 80, 200))
    _write_video(out_dir / "episode_multi_camera_wrist.mp4", color=(200, 160, 40))
    with h5py.File(out_dir / "episode_multi_camera.h5", "w") as f:
        _write_base_h5(
            f,
            episode_id="episode_multi_camera",
            language_instruction="open the drawer",
            camera_video_paths={
                "left": "episode_multi_camera_left.mp4",
                "wrist": "episode_multi_camera_wrist.mp4",
            },
            robot_profile_json=_ROBOT_PROFILE_MULTI_CAM,
        )
        _write_annotation(
            f,
            annotator="test_annotator",
            success=1.0,
            failure_description="",
            failure_category=[],
            severity="none",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_OUT = Path(__file__).resolve().parent / "samples"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=_DEFAULT_OUT,
        help=f"Destination directory (default: {_DEFAULT_OUT})",
    )
    args = parser.parse_args()
    out: Path = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    for maker in [make_unannotated, make_success, make_failure, make_multi_camera]:
        name = maker.__name__.replace("make_", "")
        print(f"  writing {name}...", end=" ", flush=True)
        maker(out)
        print("done")

    h5_files = list(out.glob("*.h5"))
    mp4_files = list(out.glob("*.mp4"))
    print(f"\nFixtures written to: {out}")
    print(f"  {len(h5_files)} HDF5 files, {len(mp4_files)} MP4 files")


if __name__ == "__main__":
    main()
