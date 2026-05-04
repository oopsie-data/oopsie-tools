"""Tests for robot profile YAML loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from oopsie_tools.utils.robot_profile.robot_profile import (
    RobotProfile,
    load_robot_profile,
    openpi_example_robot_profile_path,
)

VALID_PROFILE = {
    "policy_name": "test_policy",
    "robot_name": "test_robot",
    "is_biarm": False,
    "uses_mobile_base": False,
    "gripper_name": "test_gripper",
    "control_freq": 10,
    "camera_names": ["cam0"],
    "robot_state_keys": ["joint_position", "gripper_position"],
    "robot_state_joint_names": ["j0", "j1"],
    "action_space": ["cartesian_position", "gripper_position"],
}


def _write_profile(data: dict, tmp_dir: str) -> Path:
    p = Path(tmp_dir) / "profile.yaml"
    p.write_text(yaml.dump(data))
    return p


class TestRobotProfileValid(unittest.TestCase):
    def test_load_openpi_example_yaml(self) -> None:
        path = openpi_example_robot_profile_path()
        self.assertTrue(path.is_file(), msg=f"Missing {path}")
        profile = load_robot_profile(path)
        self.assertIsInstance(profile, RobotProfile)
        self.assertIsInstance(profile.action_space, list)
        self.assertIn("joint_velocity", profile.action_space)
        self.assertEqual(len(profile.action_joint_names or []), 7)

    def test_load_minimal_valid_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(VALID_PROFILE, tmp)
            profile = load_robot_profile(path)
            self.assertIsInstance(profile, RobotProfile)
            self.assertFalse(profile.is_biarm)


class TestRobotProfileFileErrors(unittest.TestCase):
    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_robot_profile("/nonexistent/path/profile.yaml")

    def test_non_mapping_yaml_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "profile.yaml"
            p.write_text("- just\n- a\n- list\n")
            with self.assertRaises(ValueError, msg="non-mapping YAML should raise"):
                load_robot_profile(p)


class TestRobotProfileMissingRequiredKeys(unittest.TestCase):
    def _missing_key_raises(self, key: str) -> None:
        data = {k: v for k, v in VALID_PROFILE.items() if k != key}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg=f"missing '{key}' should raise"):
                load_robot_profile(path)

    def test_missing_policy_name(self) -> None:
        self._missing_key_raises("policy_name")

    def test_missing_robot_name(self) -> None:
        self._missing_key_raises("robot_name")

    def test_missing_gripper_name(self) -> None:
        self._missing_key_raises("gripper_name")

    def test_missing_control_freq(self) -> None:
        self._missing_key_raises("control_freq")

    def test_missing_camera_names(self) -> None:
        self._missing_key_raises("camera_names")

    def test_missing_robot_state_keys(self) -> None:
        self._missing_key_raises("robot_state_keys")

    def test_missing_action_space(self) -> None:
        self._missing_key_raises("action_space")


class TestRobotProfileInvalidRobotStateKeys(unittest.TestCase):
    def test_missing_joint_position_state_key(self) -> None:
        data = {**VALID_PROFILE, "robot_state_keys": ["gripper_position"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="missing joint_position in robot_state_keys"):
                load_robot_profile(path)

    def test_missing_gripper_position_state_key(self) -> None:
        data = {**VALID_PROFILE, "robot_state_keys": ["joint_position"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="missing gripper_position in robot_state_keys"):
                load_robot_profile(path)

    def test_empty_robot_state_keys(self) -> None:
        data = {**VALID_PROFILE, "robot_state_keys": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError):
                load_robot_profile(path)


class TestRobotProfileInvalidActionSpace(unittest.TestCase):
    def test_completely_invalid_action_key(self) -> None:
        data = {**VALID_PROFILE, "action_space": ["invalid_key", "gripper_position"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="invalid action key should raise"):
                load_robot_profile(path)

    def test_no_allowed_action_key(self) -> None:
        data = {**VALID_PROFILE, "action_space": ["gripper_position"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="action_space with no ALLOWED_ACTION_SPACES key"):
                load_robot_profile(path)

    def test_missing_gripper_in_action_space(self) -> None:
        data = {**VALID_PROFILE, "action_space": ["cartesian_position"]}
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="no gripper key in action_space"):
                load_robot_profile(path)

    def test_joint_space_without_joint_names(self) -> None:
        data = {
            **VALID_PROFILE,
            "action_space": ["joint_position", "gripper_position"],
            # action_joint_names intentionally omitted
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="joint_position without action_joint_names"):
                load_robot_profile(path)

    def test_joint_velocity_without_joint_names(self) -> None:
        data = {
            **VALID_PROFILE,
            "action_space": ["joint_velocity", "gripper_position"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="joint_velocity without action_joint_names"):
                load_robot_profile(path)

    def test_mobile_base_without_base_action_key(self) -> None:
        data = {
            **VALID_PROFILE,
            "uses_mobile_base": True,
            # action_space has no base_velocity or base_position
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_profile(data, tmp)
            with self.assertRaises(ValueError, msg="mobile base without base action key"):
                load_robot_profile(path)


if __name__ == "__main__":
    unittest.main()
