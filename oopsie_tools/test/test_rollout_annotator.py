"""Tests for WebRolloutAnnotator: construction and recording delegation.

Server-dependent methods (start, wait_for_task, finish_rollout) require a
running Flask process and are not covered here.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from oopsie_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_tools.annotation_tool.rollout_annotator import WebRolloutAnnotator
from oopsie_tools.utils.robot_profile.robot_profile import RobotProfile


def _profile(**overrides) -> RobotProfile:
    defaults = dict(
        policy_name="test_policy",
        robot_name="test_robot",
        is_biarm=False,
        uses_mobile_base=False,
        gripper_name="test_gripper",
        control_freq=10,
        camera_names=["left"],
        robot_state_keys=["joint_position", "gripper_position"],
        robot_state_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
        action_space=["joint_velocity", "gripper_position"],
        action_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    )
    defaults.update(overrides)
    return RobotProfile(**defaults)


def _obs(profile: RobotProfile) -> dict:
    return {
        "robot_state": {
            "joint_position": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        },
        "image_observation": {
            cam: np.zeros((64, 64, 3), dtype=np.uint8) for cam in profile.camera_names
        },
    }


def _action(profile: RobotProfile) -> dict:
    sizes = {
        "joint_velocity": 7,
        "joint_position": 7,
        "gripper_position": 1,
        "gripper_velocity": 1,
    }
    return {
        k: np.zeros(sizes.get(k, 1), dtype=np.float32) for k in profile.action_space
    }


class TestWebRolloutAnnotatorConstruction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = _profile()
        self.annotator = WebRolloutAnnotator(
            robot_profile=self.profile,
            data_root_dir=Path(self._tmp.name),
            operator_name="  tester  ",
            annotator_name="test_annotator",
            wait_for_annotation=False,
            open_browser=False,
        )

    def tearDown(self):
        self.annotator.stop()
        self._tmp.cleanup()

    def test_operator_name_stripped(self):
        self.assertEqual(self.annotator.operator_name, "tester")

    def test_wait_for_annotation_stored(self):
        self.assertFalse(self.annotator.wait_for_annotation)

    def test_internal_recorder_is_episode_recorder(self):
        self.assertIsInstance(self.annotator._active_recorder, EpisodeRecorder)

    def test_data_root_dir_resolved(self):
        self.assertTrue(self.annotator.data_root_dir.is_absolute())

    def test_stop_is_safe_without_start(self):
        self.annotator.stop()  # _proc is None — must not raise


class TestWebRolloutAnnotatorRecording(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = _profile()
        self.annotator = WebRolloutAnnotator(
            robot_profile=self.profile,
            data_root_dir=Path(self._tmp.name),
            operator_name="tester",
            annotator_name="test_annotator",
            wait_for_annotation=False,
            open_browser=False,
        )
        self.annotator.reset_episode_recorder()

    def tearDown(self):
        self.annotator.stop()
        self._tmp.cleanup()

    def test_record_step_delegates_to_recorder(self):
        self.annotator.record_step(_obs(self.profile), _action(self.profile))
        self.assertEqual(self.annotator._active_recorder.num_steps, 1)

    def test_multiple_record_steps(self):
        for _ in range(3):
            self.annotator.record_step(_obs(self.profile), _action(self.profile))
        self.assertEqual(self.annotator._active_recorder.num_steps, 3)

    def test_reset_clears_recorder(self):
        self.annotator.record_step(_obs(self.profile), _action(self.profile))
        self.annotator.reset_episode_recorder()
        self.assertEqual(self.annotator._active_recorder.num_steps, 0)

    def test_record_step_propagates_validation_error(self):
        bad_action = {"joint_velocity": np.zeros(7)}  # missing gripper_position
        with self.assertRaises(ValueError):
            self.annotator.record_step(_obs(self.profile), bad_action)


if __name__ == "__main__":
    unittest.main()
