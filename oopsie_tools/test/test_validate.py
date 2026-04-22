"""Tests for scripts/validate_and_upload/validate.py against oopsiedata_format_v1.

Fixtures are generated once per session via conftest.py (tmp_path_factory) and
cleaned up automatically.  Per-test files use the built-in ``tmp_path`` fixture.

Sections
--------
TestReadable              – file-level guard (exists, is HDF5)
TestRequiredAttrs         – missing root attrs
TestRobotProfile          – malformed / inconsistent robot_profile JSON
TestRequiredGroups        – missing top-level or nested groups
TestTrajectoryLengths     – mismatched / zero trajectory lengths
TestVideos                – missing or too-small video files
TestValidEpisodes         – happy-path: all registered tests pass
TestValidateSessionDir    – directory-level validation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_VALIDATE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "validate_and_upload"
sys.path.insert(0, str(_VALIDATE_DIR))

from validate import validate_h5_file, validate_session_dir  # noqa: E402
from oopsie_tools.test.fixtures.make_valid import write_valid_episode  # noqa: E402


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestValidEpisodes:
    def test_unannotated_episode_passes(self, valid_episode):
        assert validate_h5_file(str(valid_episode)) is True

    def test_success_episode_passes(self, valid_success_episode):
        assert validate_h5_file(str(valid_success_episode)) is True

    def test_failure_episode_passes(self, valid_failure_episode):
        assert validate_h5_file(str(valid_failure_episode)) is True


# ---------------------------------------------------------------------------
# File-level guard
# ---------------------------------------------------------------------------


class TestReadable:
    def test_nonexistent_file_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(tmp_path / "ghost.h5"))

    def test_not_h5_raises(self, invalid_fixtures):
        with pytest.raises((AssertionError, Exception)):
            validate_h5_file(str(invalid_fixtures["invalid_not_h5"]))

    def test_empty_h5_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError):
            validate_h5_file(str(invalid_fixtures["invalid_empty_h5"]))


# ---------------------------------------------------------------------------
# Missing root attrs
# ---------------------------------------------------------------------------


class TestRequiredAttrs:
    @pytest.mark.parametrize(
        "fixture_key",
        [
            "invalid_missing_attrs",  # all attrs absent
        ],
    )
    def test_missing_attrs_raises(self, invalid_fixtures, fixture_key):
        with pytest.raises(AssertionError, match="Missing root attr"):
            validate_h5_file(str(invalid_fixtures[fixture_key]))


# ---------------------------------------------------------------------------
# robot_profile JSON validation
# ---------------------------------------------------------------------------


class TestRobotProfile:
    def test_malformed_robot_profile_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="robot_profile"):
            validate_h5_file(str(invalid_fixtures["invalid_malformed_profile"]))

    @pytest.mark.parametrize(
        "fixture_key",
        [
            "invalid_profile_missing_key",
            "invalid_profile_no_gripper",
            "invalid_profile_joint_no_names",
            "invalid_profile_unsupported_action",
            "invalid_profile_missing_rs_key",
            "invalid_profile_empty_cameras",
        ],
    )
    def test_invalid_profile_semantics_raise(self, invalid_fixtures, fixture_key):
        with pytest.raises(AssertionError):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

    def test_invalid_control_freq_zero_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="control_freq"):
            validate_h5_file(str(invalid_fixtures["invalid_control_freq_zero"]))

# ---------------------------------------------------------------------------
# Missing groups
# ---------------------------------------------------------------------------


class TestRequiredGroups:
    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            ("invalid_actions_missing", "Missing group: actions"),
            ("invalid_robot_states_missing", "Missing group: observations/robot_states"),
            # no observations/video_paths and no image_observations
            ("invalid_no_video_group", "No camera video mapping found"),
        ],
    )
    def test_missing_group_raises(self, invalid_fixtures, fixture_key, match):
        with pytest.raises(AssertionError, match=match):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

    def test_missing_robot_state_key_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="Missing observations/robot_states key"):
            validate_h5_file(str(invalid_fixtures["invalid_robot_state_missing_key"]))


# ---------------------------------------------------------------------------
# Trajectory length violations
# ---------------------------------------------------------------------------


class TestTrajectoryLengths:
    def test_mismatched_lengths_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="Inconsistent trajectory lengths"):
            validate_h5_file(str(invalid_fixtures["invalid_mismatched_steps"]))

    def test_zero_steps_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError):
            validate_h5_file(str(invalid_fixtures["invalid_zero_steps"]))

    def test_zero_trajectory_via_tmp_path(self, tmp_path):
        h5_path = write_valid_episode(tmp_path, "zero", n=0)
        with pytest.raises(AssertionError):
            validate_h5_file(str(h5_path))


# ---------------------------------------------------------------------------
# Video checks
# ---------------------------------------------------------------------------


class TestVideos:
    def test_broken_video_ref_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(invalid_fixtures["invalid_broken_video_ref"]))

    def test_array_in_video_path_raises(self, invalid_fixtures):
        with pytest.raises(AssertionError, match="does not exist"):
            validate_h5_file(str(invalid_fixtures["invalid_image_obs_float"]))

    def test_inconsistent_video_lengths_raise(self, invalid_fixtures):
        with pytest.raises(
            AssertionError,
            match="Inconsistent frame counts|Image size too small",
        ):
            validate_h5_file(str(invalid_fixtures["invalid_inconsistent_video_lengths"]))

    def test_video_length_step_mismatch_raises(self, invalid_fixtures):
        with pytest.raises(
            AssertionError,
            match="Frame count/trajectory mismatch|Image size too small",
        ):
            validate_h5_file(str(invalid_fixtures["invalid_video_length_step_mismatch"]))


# ---------------------------------------------------------------------------
# Profile/file consistency
# ---------------------------------------------------------------------------


class TestProfileFileConsistency:
    @pytest.mark.parametrize(
        "fixture_key,match",
        [
            (
                "invalid_joint_names_length_mismatch",
                "robot_state_joint_names length does not match",
            ),
            (
                "invalid_action_names_length_mismatch",
                "action_joint_names length does not match",
            ),
            (
                "invalid_profile_camera_not_in_obs",
                "Missing camera video paths",
            ),
            (
                "invalid_profile_action_not_in_recorded",
                "Missing actions key from profile.action_space",
            ),
            (
                "invalid_profile_rs_key_not_in_recorded",
                "Missing observations/robot_states key",
            ),
            (
                "invalid_multiple_promised_fields_missing",
                "Missing observations/robot_states key",
            ),
        ],
    )
    def test_profile_file_consistency_raises(
        self, invalid_fixtures, fixture_key, match
    ):
        with pytest.raises(AssertionError, match=match):
            validate_h5_file(str(invalid_fixtures[fixture_key]))

# ---------------------------------------------------------------------------
# validate_session_dir
# ---------------------------------------------------------------------------


class TestValidateSessionDir:
    def test_valid_session_passes(self, valid_session_dir):
        assert validate_session_dir(str(valid_session_dir)) == 0

    def test_nonexistent_dir_returns_1(self, tmp_path):
        assert validate_session_dir(str(tmp_path / "no_such_dir")) == 1

    def test_empty_dir_returns_1(self, tmp_path):
        assert validate_session_dir(str(tmp_path)) == 1

    def test_mixed_dir_returns_1(self, tmp_path):
        write_valid_episode(tmp_path, "good")
        (tmp_path / "bad.h5").write_text("not hdf5")
        assert validate_session_dir(str(tmp_path)) == 1

    def test_all_valid_returns_0(self, tmp_path):
        write_valid_episode(tmp_path, "ep_a")
        write_valid_episode(tmp_path, "ep_b")
        assert validate_session_dir(str(tmp_path)) == 0
