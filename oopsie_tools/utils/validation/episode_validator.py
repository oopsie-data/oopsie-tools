"""Semantic validation of EpisodeData.

All checks here operate on in-memory data (numpy arrays, VideoInfo structs)
with no file I/O.  This makes the same validation callable from:
  - The HDF5 validation pipeline (after episode_loader produces EpisodeData)
  - EpisodeRecorder pre-save (build EpisodeData from in-memory buffers first)
"""

from __future__ import annotations

import numpy as np

from oopsie_tools.utils.validation.episode_data import EpisodeData

MAX_IMAGE_SIZE = 1280
MIN_IMAGE_SIZE = 180
MIN_TRAJECTORY_LENGTH = 2
MAX_TRAJECTORY_LENGTH = 300


def validate_episode(data: EpisodeData) -> None:
    """Run all semantic checks on a loaded EpisodeData.

    Raises AssertionError with a descriptive message on the first failure.
    """
    _validate_metadata(data)
    if data.robot_profile is not None:
        _validate_profile_consistency(data)
    _validate_trajectory_lengths(data)
    _validate_video_specs(data)
    if data.annotations is not None:
        _validate_annotations(data)


# ── Individual checks ──────────────────────────────────────────────────────────

def _validate_metadata(data: EpisodeData) -> None:
    assert data.language_instruction, "language_instruction is empty"
    assert data.episode_id, "episode_id is empty"
    assert data.lab_id, "lab_id is empty"
    assert data.lab_id != "your_lab_id", "lab_id has not been changed from the placeholder value"
    assert data.operator_name, "operator_name is empty"
    assert data.control_freq > 0, "control_freq must be > 0"
    assert MIN_TRAJECTORY_LENGTH <= data.trajectory_length <= MAX_TRAJECTORY_LENGTH, (
        f"trajectory_length {data.trajectory_length} out of range "
        f"[{MIN_TRAJECTORY_LENGTH}, {MAX_TRAJECTORY_LENGTH}]"
    )


def _validate_profile_consistency(data: EpisodeData) -> None:
    """Check that observations, actions, and videos match the embedded robot profile."""
    profile = data.robot_profile
    assert profile is not None

    for key in profile.robot_state_keys:
        assert key in data.observations, f"Missing observations key required by profile: {key}. Got {list(data.observations['robot_states'].keys())}, required by profile.robot_state_keys={profile.robot_state_keys}"

    for key in profile.action_space:
        assert key in data.actions, (
            f"Missing actions key required by profile.action_space: {key}. Got {list(data.actions.keys())}, required by profile.action_space={profile.action_space}"
        )

    for cam in profile.camera_names:
        assert cam in data.videos, f"Missing video for camera required by profile: {cam}. Got {list(data.videos.keys())}, required by profile.camera_names={profile.camera_names}"

    jp_obs = data.observations.get("joint_position")
    if jp_obs is not None and jp_obs.ndim >= 2:
        assert len(profile.robot_state_joint_names) == jp_obs.shape[-1], (
            "robot_state_joint_names count does not match observations/joint_position DOF: "
            f"expected {jp_obs.shape[-1]}, got {len(profile.robot_state_joint_names)}"
        )

    if profile.action_joint_names:
        for key in ("joint_position", "joint_velocity"):
            arr = data.actions.get(key)
            if arr is not None and arr.ndim >= 2:
                assert len(profile.action_joint_names) == arr.shape[-1], (
                    f"action_joint_names count does not match actions/{key} DOF: "
                    f"expected {arr.shape[-1]}, got {len(profile.action_joint_names)}"
                )


def _validate_trajectory_lengths(data: EpisodeData) -> None:
    """All observation and action arrays must share the same trajectory length."""
    lengths: dict[str, int] = {}

    for key, arr in data.observations.items():
        if arr.ndim > 0:
            lengths[f"observations/{key}"] = arr.shape[0]

    for key, arr in data.actions.items():
        if arr.ndim > 0:
            lengths[f"actions/{key}"] = arr.shape[0]

    assert lengths, "No trajectory data found in observations or actions"

    unique = set(lengths.values())
    assert len(unique) == 1, f"Inconsistent trajectory lengths: {lengths}"

    actual_T = unique.pop()
    assert actual_T == data.trajectory_length, (
        f"trajectory_length field ({data.trajectory_length}) does not match "
        f"array shapes ({actual_T})"
    )


def _validate_video_specs(data: EpisodeData) -> None:
    """Check per-camera resolution, frame count alignment, and duration alignment."""
    assert data.videos, "No video entries found"

    T = data.trajectory_length
    frame_tolerance = max(5, int(0.1 * T))
    expected_duration = T / data.control_freq

    frame_counts: dict[str, int] = {}
    for cam, info in data.videos.items():
        assert info.width >= MIN_IMAGE_SIZE and info.height >= MIN_IMAGE_SIZE, (
            f"Video too small for camera {cam}: {info.width}x{info.height} "
            f"(min {MIN_IMAGE_SIZE}px)"
        )
        assert info.width <= MAX_IMAGE_SIZE and info.height <= MAX_IMAGE_SIZE, (
            f"Video too large for camera {cam}: {info.width}x{info.height} "
            f"(max {MAX_IMAGE_SIZE}px)"
        )
        assert abs(info.frame_count - T) <= frame_tolerance, (
            f"Frame count / trajectory mismatch for camera {cam}: "
            f"frames={info.frame_count}, trajectory={T}"
        )
        duration = info.frame_count / info.fps
        assert abs(duration - expected_duration) <= 0.5, (
            f"Video duration / control_freq mismatch for camera {cam}: "
            f"duration={duration:.2f}s, expected={expected_duration:.2f}s"
        )
        frame_counts[cam] = info.frame_count

    if len(frame_counts) > 1:
        counts = list(frame_counts.values())
        assert max(counts) - min(counts) <= 1, (
            f"Inconsistent frame counts across cameras: {frame_counts}"
        )


def _validate_annotations(data: EpisodeData) -> None:
    """Each annotator subgroup must have a numeric success score in [0.0, 1.0]."""
    assert data.annotations, "annotations dict is empty"

    for annotator, attrs in data.annotations.items():
        assert "success" in attrs, (
            f"episode_annotations/{annotator} is missing 'success' — "
            "episode has not been fully annotated yet"
        )
        try:
            success = float(attrs["success"])
        except (TypeError, ValueError):
            raise AssertionError(
                f"episode_annotations/{annotator}/success is not numeric: "
                f"{attrs['success']!r}"
            )
        assert not np.isnan(success), (
            f"episode_annotations/{annotator}/success is NaN — "
            "episode has not been fully annotated yet"
        )
        assert 0.0 <= success <= 1.0, (
            f"episode_annotations/{annotator}/success out of range [0.0, 1.0]: {success}"
        )
