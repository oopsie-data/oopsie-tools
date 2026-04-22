"""Tests for EpisodeRecorder: buffering, validation, and HDF5 output."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from oopsie_tools.annotation_tool.episode_recorder import (
    EpisodeRecorder,
    write_mp4,
)
from oopsie_tools.utils.robot_profile import RobotProfile


def _profile(**overrides) -> RobotProfile:
    defaults = dict(
        policy_name="test_policy",
        robot_name="test_robot",
        is_biarm=False,
        gripper_name="test_gripper",
        control_freq=10,
        camera_names=["left", "wrist"],
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


def _save_data(recorder: EpisodeRecorder, cam_names: list[str]) -> dict:
    return {
        "language_instruction": "pick up the cup",
        "metadata": {
            "episode_id": recorder.save_fname,
            "operator_name": "tester",
        },
        "video_paths": {cam: f"/tmp/{cam}.mp4" for cam in cam_names},
    }


class TestEpisodeRecorderInit(unittest.TestCase):
    def test_session_dir_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(robot_profile=_profile(), data_root_dir=tmp)
            self.assertTrue(recorder.session_dir.is_dir())

    def test_resume_session_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(
                robot_profile=_profile(),
                data_root_dir=tmp,
                resume_session_name="my_session",
            )
            self.assertEqual(recorder.session_name, "my_session")

    def test_initial_num_steps_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = EpisodeRecorder(robot_profile=_profile(), data_root_dir=tmp)
            self.assertEqual(recorder.num_steps, 0)


class TestEpisodeRecorderRecordStep(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = _profile()
        self.recorder = EpisodeRecorder(
            robot_profile=self.profile, data_root_dir=self._tmp.name
        )
        self.recorder.reset_episode_recorder()

    def tearDown(self):
        self._tmp.cleanup()

    def test_record_step_increments_count(self):
        self.recorder.record_step(_obs(self.profile), _action(self.profile))
        self.assertEqual(self.recorder.num_steps, 1)

    def test_multiple_steps_accumulate(self):
        for _ in range(5):
            self.recorder.record_step(_obs(self.profile), _action(self.profile))
        self.assertEqual(self.recorder.num_steps, 5)

    def test_reset_clears_steps(self):
        self.recorder.record_step(_obs(self.profile), _action(self.profile))
        self.recorder.reset_episode_recorder()
        self.assertEqual(self.recorder.num_steps, 0)

    def test_rejects_non_dict_observation(self):
        with self.assertRaises(ValueError):
            self.recorder.record_step("not a dict", _action(self.profile))

    def test_rejects_missing_robot_state_key(self):
        obs = _obs(self.profile)
        del obs["robot_state"]
        with self.assertRaises(ValueError):
            self.recorder.record_step(obs, _action(self.profile))

    def test_rejects_missing_image_observation_key(self):
        obs = _obs(self.profile)
        del obs["image_observation"]
        with self.assertRaises(ValueError):
            self.recorder.record_step(obs, _action(self.profile))

    def test_rejects_missing_camera(self):
        obs = _obs(self.profile)
        del obs["image_observation"]["left"]
        with self.assertRaises(ValueError):
            self.recorder.record_step(obs, _action(self.profile))

    def test_rejects_missing_robot_state_component(self):
        obs = _obs(self.profile)
        del obs["robot_state"]["joint_position"]
        with self.assertRaises(ValueError):
            self.recorder.record_step(obs, _action(self.profile))

    def test_rejects_empty_action(self):
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), {})

    def test_rejects_unrecognized_action_key(self):
        action = {**_action(self.profile), "bad_key": np.zeros(1)}
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), action)

    def test_rejects_mismatched_action_keys(self):
        with self.assertRaises(ValueError):
            self.recorder.record_step(
                _obs(self.profile), {"joint_velocity": np.zeros(7)}
            )

    def test_rejects_none_action_values(self):
        action = {k: None for k in self.profile.action_space}
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), action)

    def test_rejects_non_dict_action(self):
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), np.zeros(8))


class TestEpisodeRecorderSave(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = _profile()
        self.recorder = EpisodeRecorder(
            robot_profile=self.profile, data_root_dir=self._tmp.name
        )
        self.recorder.reset_episode_recorder()

    def tearDown(self):
        self._tmp.cleanup()

    def _record_n(self, n: int = 3) -> None:
        for _ in range(n):
            self.recorder.record_step(_obs(self.profile), _action(self.profile))

    def test_save_returns_h5_path(self):
        self._record_n()
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        self.assertIsInstance(h5_path, Path)
        self.assertEqual(h5_path.suffix, ".h5")

    def test_save_file_exists(self):
        self._record_n()
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        self.assertTrue(h5_path.exists())

    def test_save_h5_attrs(self):
        self._record_n()
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        with h5py.File(h5_path, "r") as f:
            self.assertEqual(f.attrs["language_instruction"], "pick up the cup")
            self.assertEqual(f.attrs["schema"], "oopsiedata_format_v1")

    def test_save_h5_observations_and_robot_states(self):
        self._record_n()
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        with h5py.File(h5_path, "r") as f:
            self.assertIn("observations", f)
            self.assertIn("robot_states", f["observations"])
            self.assertIn("joint_position", f["observations/robot_states"])

    def test_save_h5_robot_state_timestep_count(self):
        self._record_n(4)
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        with h5py.File(h5_path, "r") as f:
            self.assertEqual(f["observations/robot_states/joint_position"].shape[0], 4)

    def test_save_h5_actions_group(self):
        self._record_n()
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        with h5py.File(h5_path, "r") as f:
            self.assertIn("actions", f)

    def test_save_raises_without_steps(self):
        with self.assertRaises(ValueError):
            self.recorder.save(_save_data(self.recorder, self.profile.camera_names))

    def test_save_in_session_dir(self):
        self._record_n()
        h5_path = self.recorder.save(
            _save_data(self.recorder, self.profile.camera_names)
        )
        self.assertEqual(h5_path.parent, self.recorder.session_dir)


class TestWriteMp4Validation(unittest.TestCase):
    def test_wrong_ndim_raises(self):
        frames = np.zeros((64, 64, 3), dtype=np.uint8)  # missing time dimension
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError, msg="3-D array should raise"):
                write_mp4(Path(tmp) / "out.mp4", frames, fps=10.0)

    def test_wrong_channel_count_raises(self):
        frames = np.zeros((4, 64, 64, 4), dtype=np.uint8)  # RGBA instead of RGB
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError, msg="RGBA frames should raise"):
                write_mp4(Path(tmp) / "out.mp4", frames, fps=10.0)

    def test_zero_frames_raises(self):
        frames = np.zeros((0, 64, 64, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError, msg="zero-frame array should raise"):
                write_mp4(Path(tmp) / "out.mp4", frames, fps=10.0)

    def test_2d_array_raises(self):
        frames = np.zeros((64, 64), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_mp4(Path(tmp) / "out.mp4", frames, fps=10.0)


class TestRecordStepActionBreaking(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.profile = _profile()
        self.recorder = EpisodeRecorder(
            robot_profile=self.profile, data_root_dir=self._tmp.name
        )
        self.recorder.reset_episode_recorder()

    def tearDown(self):
        self._tmp.cleanup()

    def test_rejects_partial_none_action_value(self):
        """One None among otherwise valid keys should still raise."""
        action = _action(self.profile)
        first_key = next(iter(action))
        action[first_key] = None
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), action)

    def test_rejects_action_superset_of_profile(self):
        """Extra valid action key beyond profile's action_space should raise."""
        action = _action(self.profile)
        action["cartesian_position"] = np.zeros(7, dtype=np.float32)
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), action)

    def test_rejects_action_subset_of_profile(self):
        """Providing only one of the required profile action keys should raise."""
        action = {"joint_velocity": np.zeros(7, dtype=np.float32)}
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), action)

    def test_rejects_action_with_wrong_but_valid_global_keys(self):
        """Globally-valid keys that don't match the profile's action_space should raise."""
        action = {
            "cartesian_position": np.zeros(7, dtype=np.float32),
            "gripper_position": np.zeros(1, dtype=np.float32),
        }
        with self.assertRaises(ValueError):
            self.recorder.record_step(_obs(self.profile), action)

    def test_rejects_extra_camera_not_in_profile(self):
        """image_observation with extra camera beyond profile is fine, but missing one raises."""
        obs = _obs(self.profile)
        del obs["image_observation"]["wrist"]
        with self.assertRaises(ValueError):
            self.recorder.record_step(obs, _action(self.profile))

    def test_rejects_extra_robot_state_key_missing_required(self):
        """robot_state with a different key than required should raise."""
        obs = _obs(self.profile)
        obs["robot_state"] = {"unexpected_key": np.zeros(7)}
        with self.assertRaises(ValueError):
            self.recorder.record_step(obs, _action(self.profile))


if __name__ == "__main__":
    unittest.main()
