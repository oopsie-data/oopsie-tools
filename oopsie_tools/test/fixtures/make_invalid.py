#!/usr/bin/env python3
"""Generate intentionally malformed HDF5 episodes for negative testing.

Each fixture targets one specific failure mode so tests can verify that the
annotator server, analysis scripts, and any other reader either degrade
gracefully (return defaults / empty results) or raise the expected error,
rather than crashing silently or producing wrong output.

Scenarios are grouped by which layer they exercise:

A – Annotator server file-level guards
    invalid_not_h5              .h5 extension but file is plain text
    invalid_empty_h5            valid HDF5 with no groups or attrs
    invalid_missing_attrs       correct structure, all root attrs absent
    invalid_broken_video_ref    image_observations/front → nonexistent file
    invalid_no_video_group      no image_observations and no <stem>_*.mp4 sibling
    invalid_annotation_dataset  episode_annotations is a scalar dataset, not a group
    invalid_taxonomy_not_json   annotation subgroup has malformed taxonomy JSON

B – HDF5 schema / data violations (structure)
    invalid_mismatched_steps    robot_states 20 steps, actions 5 steps
    invalid_zero_steps          robot_states and actions have 0 rows
    invalid_actions_missing     actions group entirely absent
    invalid_robot_states_missing observations/robot_states group absent
    invalid_image_obs_float     image_observations/front stores a float array, not a path string

C – Tensor shape / dtype violations
    invalid_joint_pos_wrong_dof     joint_position is (20, 3) not (20, 7)
    invalid_gripper_pos_wrong_dof   gripper_position is (20, 3) not (20, 1)
    invalid_joint_vel_wrong_dof     joint_velocity action is (20, 3) not (20, 7)
    invalid_robot_state_wrong_dtype joint_position stored as uint8 not float64
    invalid_robot_state_extra_key   robot_states has undeclared extra key
    invalid_robot_state_missing_key gripper_position absent from robot_states group

D – Embedded robot_profile JSON violations
    invalid_malformed_profile       robot_profile attr is not valid JSON
    invalid_profile_missing_key     required key policy_name absent from profile JSON
    invalid_profile_no_gripper      action_space has no gripper key
    invalid_profile_joint_no_names  joint action_space but action_joint_names absent
    invalid_profile_unsupported_action action_space contains unknown key "hand_position"
    invalid_profile_empty_cameras   camera_names is empty list
    invalid_profile_missing_rs_key  robot_state_keys missing required "joint_position"
    invalid_profile_biarm_mismatch  is_biarm=True but tensors are single-arm (7-dim)

E – Cross-consistency violations
    invalid_joint_names_length_mismatch  robot_state_joint_names has 7 entries, tensor has 5 cols
    invalid_action_names_length_mismatch action_joint_names has 7 entries, tensor has 6 cols
    invalid_profile_camera_not_in_obs    profile declares wrist camera, image_observations omits it
    invalid_profile_action_not_in_recorded profile declares joint_velocity, actions group omits it
    invalid_profile_rs_key_not_in_recorded profile declares eef_pos, robot_states omits it
    invalid_multiple_promised_fields_missing three fields promised by profile all absent
    invalid_control_freq_zero           control_freq=0 causes divide-by-zero in frame math
    invalid_inconsistent_video_lengths  front MP4 has 10 frames, wrist MP4 has 30 frames
    invalid_video_length_step_mismatch  MP4 has 10 frames, robot_states has 100 rows

Usage
-----
    uv run python -m oopsie_tools.test.fixtures.make_invalid [OUTPUT_DIR]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable, Tuple

import h5py
import imageio
import numpy as np

_VideoWriter = Callable[[Path, Tuple[int, int, int], int], None]

_STR_DTYPE = h5py.string_dtype(encoding="utf-8")

_VALID_ROBOT_PROFILE = {
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


def _write_video(path: Path, color: tuple, n_frames: int = 10) -> None:
    frames = np.full((n_frames, 64, 64, 3), color, dtype=np.uint8)
    with imageio.get_writer(
        str(path), format="FFMPEG", mode="I", fps=10, codec="libx264",
        output_params=["-crf", "28"],
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


# ---------------------------------------------------------------------------
# Shared builders
# ---------------------------------------------------------------------------


def _base_attrs(
    f: h5py.File,
    episode_id: str = "test_episode",
    robot_profile: dict | str | None = None,
) -> None:
    f.attrs["schema"] = "oopsiedata_format_v1"
    f.attrs["episode_id"] = episode_id
    if robot_profile is None:
        robot_profile = _VALID_ROBOT_PROFILE
    f.attrs["robot_profile"] = (
        robot_profile if isinstance(robot_profile, str) else json.dumps(robot_profile)
    )
    f.attrs["language_instruction"] = "pick up the block"
    f.attrs["operator_name"] = "test_operator"
    f.attrs["lab_id"] = "test_lab"
    f.attrs["timestamp"] = time.time()


def _robot_states(f: h5py.File, n: int = 20, joint_dof: int = 7, gripper_dof: int = 1) -> None:
    rs = f.require_group("observations/robot_states")
    rs.create_dataset("joint_position", data=np.zeros((n, joint_dof), dtype=np.float64))
    rs.create_dataset("gripper_position", data=np.zeros((n, gripper_dof), dtype=np.float64))


def _actions(f: h5py.File, n: int = 20, joint_dof: int = 7) -> None:
    ag = f.require_group("actions")
    ag.create_dataset("joint_velocity", data=np.zeros((n, joint_dof), dtype=np.float64))
    ag.create_dataset("gripper_position", data=np.zeros((n, 1), dtype=np.float64))
    for key in (
        "cartesian_position", "cartesian_velocity", "joint_position",
        "base_position", "base_velocity", "gripper_velocity", "gripper_binary",
    ):
        ag.create_dataset(key, data=h5py.Empty(dtype=np.float64))


def _video_paths(f: h5py.File, cam: str, rel_path: str) -> None:
    vp = f.require_group("observations/video_paths")
    vp.create_dataset(cam, data=rel_path, dtype=_STR_DTYPE)


def _full_valid_episode(f: h5py.File, episode_id: str, cam_path: str = "front.mp4") -> None:
    _base_attrs(f, episode_id)
    _robot_states(f)
    _actions(f)
    _video_paths(f, "front", cam_path)


# ---------------------------------------------------------------------------
# A – Annotator server file-level guards
# ---------------------------------------------------------------------------


def make_not_h5(out_dir: Path) -> None:
    (out_dir / "invalid_not_h5.h5").write_text("this is not an HDF5 file\n")


def make_empty_h5(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_empty_h5.h5", "w"):
        pass


def make_missing_attrs(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_missing_attrs.h5", "w") as f:
        f.attrs["schema"] = "oopsiedata_format_v1"
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_missing_attrs_front.mp4")


def make_broken_video_ref(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_broken_video_ref.h5", "w") as f:
        _base_attrs(f, "invalid_broken_video_ref")
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "does_not_exist.mp4")


def make_no_video_group(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_no_video_group.h5", "w") as f:
        _base_attrs(f, "invalid_no_video_group")
        _robot_states(f)
        _actions(f)


def make_annotation_dataset(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_annotation_dataset.h5", "w") as f:
        _full_valid_episode(f, "invalid_annotation_dataset", "invalid_annotation_dataset_front.mp4")
        f.create_dataset("episode_annotations", data="not a group", dtype=_STR_DTYPE)


def make_taxonomy_not_json(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_taxonomy_not_json.h5", "w") as f:
        _full_valid_episode(f, "invalid_taxonomy_not_json", "invalid_taxonomy_not_json_front.mp4")
        ag = f.require_group("episode_annotations/test_annotator")
        ag.attrs["schema"] = "oopsie_failure_taxonomy_v1"
        ag.attrs["source"] = "human"
        ag.attrs["timestamp"] = "2026-04-21T10:00:00+00:00"
        ag.attrs["success"] = 0.0
        ag.attrs["failure_description"] = "Robot dropped the object."
        ag.attrs["taxonomy_schema"] = "oopsiedata_taxonomy_schema_v1"
        ag.attrs["taxonomy"] = "{ this is: not: valid json !!!"
        ag.attrs["additional_notes"] = ""


# ---------------------------------------------------------------------------
# B – HDF5 schema / data violations (structure)
# ---------------------------------------------------------------------------


def make_mismatched_steps(out_dir: Path) -> None:
    _write_video(out_dir / "invalid_mismatched_steps_front.mp4", (100, 100, 200))
    with h5py.File(out_dir / "invalid_mismatched_steps.h5", "w") as f:
        _base_attrs(f, "invalid_mismatched_steps")
        _robot_states(f, n=20)
        _actions(f, n=5)
        _video_paths(f, "front", "invalid_mismatched_steps_front.mp4")


def make_zero_steps(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_zero_steps.h5", "w") as f:
        _base_attrs(f, "invalid_zero_steps")
        _robot_states(f, n=0)
        _actions(f, n=0)
        _video_paths(f, "front", "invalid_zero_steps_front.mp4")


def make_actions_missing(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_actions_missing.h5", "w") as f:
        _base_attrs(f, "invalid_actions_missing")
        _robot_states(f)
        _video_paths(f, "front", "invalid_actions_missing_front.mp4")


def make_robot_states_missing(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_robot_states_missing.h5", "w") as f:
        _base_attrs(f, "invalid_robot_states_missing")
        _actions(f)
        _video_paths(f, "front", "invalid_robot_states_missing_front.mp4")


def make_image_obs_float(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_image_obs_float.h5", "w") as f:
        _base_attrs(f, "invalid_image_obs_float")
        _robot_states(f)
        _actions(f)
        # Store a float array instead of a path string. The loader will decode it
        # to a garbage path and fail with "does not exist".
        vp = f.require_group("observations/video_paths")
        vp.create_dataset("front", data=np.zeros((10, 64, 64, 3), dtype=np.uint8))


# ---------------------------------------------------------------------------
# C – Tensor shape / dtype violations
# ---------------------------------------------------------------------------


def make_joint_pos_wrong_dof(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_joint_pos_wrong_dof.h5", "w") as f:
        _base_attrs(f, "invalid_joint_pos_wrong_dof")
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 3), dtype=np.float64))
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_joint_pos_wrong_dof_front.mp4")


def make_gripper_pos_wrong_dof(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_gripper_pos_wrong_dof.h5", "w") as f:
        _base_attrs(f, "invalid_gripper_pos_wrong_dof")
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 7), dtype=np.float64))
        rs.create_dataset("gripper_position", data=np.zeros((20, 3), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_gripper_pos_wrong_dof_front.mp4")


def make_joint_vel_wrong_dof(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_joint_vel_wrong_dof.h5", "w") as f:
        _base_attrs(f, "invalid_joint_vel_wrong_dof")
        _robot_states(f)
        ag = f.require_group("actions")
        ag.create_dataset("joint_velocity", data=np.zeros((20, 3), dtype=np.float64))
        ag.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        for key in (
            "cartesian_position", "cartesian_velocity", "joint_position",
            "base_position", "base_velocity", "gripper_velocity", "gripper_binary",
        ):
            ag.create_dataset(key, data=h5py.Empty(dtype=np.float64))
        _video_paths(f, "front", "invalid_joint_vel_wrong_dof_front.mp4")


def make_robot_state_wrong_dtype(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_robot_state_wrong_dtype.h5", "w") as f:
        _base_attrs(f, "invalid_robot_state_wrong_dtype")
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.full((20, 7), 128, dtype=np.uint8))
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_robot_state_wrong_dtype_front.mp4")


def make_robot_state_extra_key(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_robot_state_extra_key.h5", "w") as f:
        _base_attrs(f, "invalid_robot_state_extra_key")
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 7), dtype=np.float64))
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        rs.create_dataset("velocity_hack", data=np.zeros((20, 7), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_robot_state_extra_key_front.mp4")


def make_robot_state_missing_key(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_robot_state_missing_key.h5", "w") as f:
        _base_attrs(f, "invalid_robot_state_missing_key")
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 7), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_robot_state_missing_key_front.mp4")


# ---------------------------------------------------------------------------
# D – Embedded robot_profile JSON violations
# ---------------------------------------------------------------------------


def make_malformed_profile(out_dir: Path) -> None:
    with h5py.File(out_dir / "invalid_malformed_profile.h5", "w") as f:
        _base_attrs(f, "invalid_malformed_profile", robot_profile="{ broken json ...")
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_malformed_profile_front.mp4")


def make_profile_missing_key(out_dir: Path) -> None:
    profile = {k: v for k, v in _VALID_ROBOT_PROFILE.items() if k != "policy_name"}
    with h5py.File(out_dir / "invalid_profile_missing_key.h5", "w") as f:
        _base_attrs(f, "invalid_profile_missing_key", robot_profile=profile)
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_profile_missing_key_front.mp4")


def make_profile_no_gripper(out_dir: Path) -> None:
    profile = {**_VALID_ROBOT_PROFILE, "action_space": ["joint_velocity"]}
    with h5py.File(out_dir / "invalid_profile_no_gripper.h5", "w") as f:
        _base_attrs(f, "invalid_profile_no_gripper", robot_profile=profile)
        _robot_states(f)
        ag = f.require_group("actions")
        ag.create_dataset("joint_velocity", data=np.zeros((20, 7), dtype=np.float64))
        _video_paths(f, "front", "invalid_profile_no_gripper_front.mp4")


def make_profile_joint_no_names(out_dir: Path) -> None:
    profile = {**_VALID_ROBOT_PROFILE, "action_joint_names": None}
    with h5py.File(out_dir / "invalid_profile_joint_no_names.h5", "w") as f:
        _base_attrs(f, "invalid_profile_joint_no_names", robot_profile=profile)
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_profile_joint_no_names_front.mp4")


def make_profile_unsupported_action(out_dir: Path) -> None:
    profile = {
        **_VALID_ROBOT_PROFILE,
        "action_space": ["hand_position", "gripper_position"],
    }
    with h5py.File(out_dir / "invalid_profile_unsupported_action.h5", "w") as f:
        _base_attrs(f, "invalid_profile_unsupported_action", robot_profile=profile)
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_profile_unsupported_action_front.mp4")


def make_profile_empty_cameras(out_dir: Path) -> None:
    profile = {**_VALID_ROBOT_PROFILE, "camera_names": []}
    with h5py.File(out_dir / "invalid_profile_empty_cameras.h5", "w") as f:
        _base_attrs(f, "invalid_profile_empty_cameras", robot_profile=profile)
        _robot_states(f)
        _actions(f)


def make_profile_missing_rs_key(out_dir: Path) -> None:
    profile = {
        **_VALID_ROBOT_PROFILE,
        "robot_state_keys": ["gripper_position"],
        "robot_state_joint_names": [],
    }
    with h5py.File(out_dir / "invalid_profile_missing_rs_key.h5", "w") as f:
        _base_attrs(f, "invalid_profile_missing_rs_key", robot_profile=profile)
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_profile_missing_rs_key_front.mp4")


def make_profile_biarm_mismatch(out_dir: Path) -> None:
    profile = {**_VALID_ROBOT_PROFILE, "is_biarm": True}
    with h5py.File(out_dir / "invalid_profile_biarm_mismatch.h5", "w") as f:
        _base_attrs(f, "invalid_profile_biarm_mismatch", robot_profile=profile)
        _robot_states(f, joint_dof=7)
        _actions(f, joint_dof=7)
        _video_paths(f, "front", "invalid_profile_biarm_mismatch_front.mp4")


# ---------------------------------------------------------------------------
# E – Cross-consistency violations
# ---------------------------------------------------------------------------


def make_joint_names_length_mismatch(out_dir: Path) -> None:
    """robot_state_joint_names declares 7 names but joint_position tensor has 5 columns."""
    _write_video(out_dir / "invalid_joint_names_length_mismatch_front.mp4", (200, 100, 100))
    profile = {
        **_VALID_ROBOT_PROFILE,
        "robot_state_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    }
    with h5py.File(out_dir / "invalid_joint_names_length_mismatch.h5", "w") as f:
        _base_attrs(f, "invalid_joint_names_length_mismatch", robot_profile=profile)
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 5), dtype=np.float64))
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_joint_names_length_mismatch_front.mp4")


def make_action_names_length_mismatch(out_dir: Path) -> None:
    """action_joint_names declares 7 names but joint_velocity action has 6 columns."""
    _write_video(out_dir / "invalid_action_names_length_mismatch_front.mp4", (100, 200, 100))
    profile = {
        **_VALID_ROBOT_PROFILE,
        "action_joint_names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    }
    with h5py.File(out_dir / "invalid_action_names_length_mismatch.h5", "w") as f:
        _base_attrs(f, "invalid_action_names_length_mismatch", robot_profile=profile)
        _robot_states(f)
        ag = f.require_group("actions")
        ag.create_dataset("joint_velocity", data=np.zeros((20, 6), dtype=np.float64))
        ag.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        for key in (
            "cartesian_position", "cartesian_velocity", "joint_position",
            "base_position", "base_velocity", "gripper_velocity", "gripper_binary",
        ):
            ag.create_dataset(key, data=h5py.Empty(dtype=np.float64))
        _video_paths(f, "front", "invalid_action_names_length_mismatch_front.mp4")


def make_profile_camera_not_in_obs(out_dir: Path) -> None:
    """Profile declares two cameras but image_observations only has one."""
    profile = {**_VALID_ROBOT_PROFILE, "camera_names": ["front", "wrist"]}
    with h5py.File(out_dir / "invalid_profile_camera_not_in_obs.h5", "w") as f:
        _base_attrs(f, "invalid_profile_camera_not_in_obs", robot_profile=profile)
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_profile_camera_not_in_obs_front.mp4")


def make_profile_action_not_in_recorded(out_dir: Path) -> None:
    """Profile declares joint_velocity + gripper_position but only gripper_position was recorded."""
    with h5py.File(out_dir / "invalid_profile_action_not_in_recorded.h5", "w") as f:
        _base_attrs(f, "invalid_profile_action_not_in_recorded")
        _robot_states(f)
        ag = f.require_group("actions")
        ag.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _video_paths(f, "front", "invalid_profile_action_not_in_recorded_front.mp4")


def make_profile_rs_key_not_in_recorded(out_dir: Path) -> None:
    """Profile declares eef_cartesian_position in robot_state_keys but it's absent in the file."""
    profile = {
        **_VALID_ROBOT_PROFILE,
        "robot_state_keys": ["joint_position", "gripper_position", "eef_cartesian_position"],
    }
    with h5py.File(out_dir / "invalid_profile_rs_key_not_in_recorded.h5", "w") as f:
        _base_attrs(f, "invalid_profile_rs_key_not_in_recorded", robot_profile=profile)
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 7), dtype=np.float64))
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _actions(f)
        _video_paths(f, "front", "invalid_profile_rs_key_not_in_recorded_front.mp4")


def make_multiple_promised_fields_missing(out_dir: Path) -> None:
    """Three fields promised by profile (eef state, joint_position action, wrist camera) all absent.

    This catches validators that only check one missing field at a time and stop,
    potentially masking the remaining violations.
    """
    profile = {
        **_VALID_ROBOT_PROFILE,
        "camera_names": ["front", "wrist"],
        "robot_state_keys": ["joint_position", "gripper_position", "eef_cartesian_position"],
        "action_space": ["joint_position", "gripper_position", "gripper_velocity"],
    }
    with h5py.File(out_dir / "invalid_multiple_promised_fields_missing.h5", "w") as f:
        _base_attrs(f, "invalid_multiple_promised_fields_missing", robot_profile=profile)
        rs = f.require_group("observations/robot_states")
        rs.create_dataset("joint_position", data=np.zeros((20, 7), dtype=np.float64))
        rs.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        ag = f.require_group("actions")
        ag.create_dataset("gripper_position", data=np.zeros((20, 1), dtype=np.float64))
        _video_paths(f, "front", "invalid_multiple_promised_fields_missing_front.mp4")


def make_control_freq_zero(out_dir: Path) -> None:
    """control_freq=0 causes divide-by-zero in any frame-to-timestep calculation."""
    profile = {**_VALID_ROBOT_PROFILE, "control_freq": 0}
    with h5py.File(out_dir / "invalid_control_freq_zero.h5", "w") as f:
        _base_attrs(f, "invalid_control_freq_zero", robot_profile=profile)
        _robot_states(f)
        _actions(f)
        _video_paths(f, "front", "invalid_control_freq_zero_front.mp4")


def make_inconsistent_video_lengths(out_dir: Path, video_writer: _VideoWriter) -> None:
    """front MP4 has 10 frames, wrist MP4 has 30 — they cannot represent the same episode."""
    video_writer(out_dir / "invalid_inconsistent_video_lengths_front.mp4", (80, 120, 200), 10)
    video_writer(out_dir / "invalid_inconsistent_video_lengths_wrist.mp4", (200, 80, 80), 30)

    profile = {**_VALID_ROBOT_PROFILE, "camera_names": ["front", "wrist"]}
    with h5py.File(out_dir / "invalid_inconsistent_video_lengths.h5", "w") as f:
        _base_attrs(f, "invalid_inconsistent_video_lengths", robot_profile=profile)
        _robot_states(f, n=20)
        _actions(f, n=20)
        vp = f.require_group("observations/video_paths")
        vp.create_dataset("front", data="invalid_inconsistent_video_lengths_front.mp4", dtype=_STR_DTYPE)
        vp.create_dataset("wrist", data="invalid_inconsistent_video_lengths_wrist.mp4", dtype=_STR_DTYPE)


def make_video_length_step_mismatch(out_dir: Path, video_writer: _VideoWriter) -> None:
    """MP4 has 10 frames but robot_states has 100 rows — video was truncated."""
    video_writer(out_dir / "invalid_video_length_step_mismatch_front.mp4", (80, 200, 120), 10)
    with h5py.File(out_dir / "invalid_video_length_step_mismatch.h5", "w") as f:
        _base_attrs(f, "invalid_video_length_step_mismatch")
        _robot_states(f, n=100)
        _actions(f, n=100)
        _video_paths(f, "front", "invalid_video_length_step_mismatch_front.mp4")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_DEFAULT_OUT = Path(__file__).resolve().parent / "samples"

_MAKERS_SIMPLE = [
    make_not_h5, make_empty_h5, make_missing_attrs, make_broken_video_ref,
    make_no_video_group, make_annotation_dataset, make_taxonomy_not_json,
    make_mismatched_steps, make_zero_steps, make_actions_missing,
    make_robot_states_missing, make_image_obs_float,
    make_joint_pos_wrong_dof, make_gripper_pos_wrong_dof, make_joint_vel_wrong_dof,
    make_robot_state_wrong_dtype, make_robot_state_extra_key, make_robot_state_missing_key,
    make_malformed_profile, make_profile_missing_key, make_profile_no_gripper,
    make_profile_joint_no_names, make_profile_unsupported_action, make_profile_empty_cameras,
    make_profile_missing_rs_key, make_profile_biarm_mismatch,
    make_joint_names_length_mismatch, make_action_names_length_mismatch,
    make_profile_camera_not_in_obs, make_profile_action_not_in_recorded,
    make_profile_rs_key_not_in_recorded, make_multiple_promised_fields_missing,
    make_control_freq_zero,
]

_MAKERS_WITH_VIDEO = [
    make_inconsistent_video_lengths,
    make_video_length_step_mismatch,
]


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

    for maker in _MAKERS_SIMPLE:
        name = maker.__name__.replace("make_", "")
        print(f"  writing {name}...", end=" ", flush=True)
        maker(out)
        print("done")

    for maker in _MAKERS_WITH_VIDEO:
        name = maker.__name__.replace("make_", "")
        print(f"  writing {name}...", end=" ", flush=True)
        maker(out, _write_video)
        print("done")

    h5_files = list(out.glob("invalid_*.h5"))
    print(f"\nInvalid fixtures written to: {out}")
    print(f"  {len(h5_files)} invalid HDF5 files")


if __name__ == "__main__":
    main()
