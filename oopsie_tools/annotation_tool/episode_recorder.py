"""Episode recorder for saving robot evaluation data in HDF5 format."""

from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any
from moviepy.editor import ImageSequenceClip

import h5py
import imageio
import numpy as np
import datetime
import yaml
from oopsie_tools.utils.robot_profile import RobotProfile, robot_profile_to_json
from oopsie_tools.utils.rotation_utils import ActionQuatConversion

REQUIRED_OBSERVATION_KEYS = ["robot_state", "image_observation"]

VALID_ACTION_KEYS = {
    "cartesian_position",
    "cartesian_velocity",
    "joint_position",
    "joint_velocity",
    "base_position",
    "base_velocity",
    "gripper_velocity",
    "gripper_position",
    "gripper_binary",
}


def write_mp4(video_path: Path, frames: np.ndarray, fps: float) -> None:
    """Write RGB frames to an MP4 file.

    Args:
        video_path (Path): Destination path for the MP4 file.
        frames (np.ndarray): Video frames with shape ``(T, H, W, 3)``.
        fps (float): Output video frame rate.

    Raises:
        ValueError: If ``frames`` does not have shape ``(T, H, W, 3)`` or
            contains zero timesteps.
    """
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


class EpisodeRecorder:
    """Record rollout observations and persist them in HDF5 format.

    The recorder stores proprioception, optional video references, and
    annotation metadata under the robotic failure upload schema.
    """

    def __init__(
        self,
        robot_profile: RobotProfile,
        data_root_dir: Path | str,
        resume_session_name: str | None = None,
        operator_name: str | None = None,
    ) -> None:
        """Initialize a recorder instance.

        Args:
            robot_profile (RobotProfile): Robot profile.
            data_root_dir (str): Base output directory for saved artifacts.
            session_name (str | None): Optional unique session name

        Raises:
            ValueError: If ``data_root_dir`` is not a valid directory.
        """
        self.data_root_dir = Path(data_root_dir)
        self.session_name = (
            resume_session_name
            if resume_session_name is not None
            else datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.session_dir = self.data_root_dir / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.operator_name = operator_name

        self.robot_profile = robot_profile
        self.camera_names = robot_profile.camera_names

        self.quat_conversion = (
            ActionQuatConversion(
                self.robot_profile.get_rot_option(),
                is_biarm=self.robot_profile.is_biarm,
            )
            if self.robot_profile.orientation_representation
            else None
        )
        self.robot_state_quat_conversion = (
            ActionQuatConversion(
                self.robot_profile.get_robot_state_rot_option(),
                is_biarm=self.robot_profile.is_biarm,
            )
            if self.robot_profile.robot_state_orientation_representation
            else None
        )
        self.frames: dict[str, list[np.ndarray]] = {
            cam: [] for cam in self.camera_names
        }
        # Timestep data buffer
        self.timesteps: list[dict[str, Any]] = []

        # Read lab_id from configs/contributor_config.yaml
        try:
            config_path = (
                Path(__file__).resolve().parent.parent.parent
                / "configs"
                / "contributor_config.yaml"
            )
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                self.lab_id = config.get("lab_id", "").strip()
                if not self.lab_id:
                    raise ValueError(
                        "lab_id must be set in configs/contributor_config.yaml"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Could not read lab_id from configs/contributor_config.yaml: {e}"
            )

    def reset_episode_recorder(self) -> None:
        """Reset the buffers for a new episode."""
        ts = datetime.datetime.now()
        self.timestamp = ts.timestamp()
        self.save_fname = f"{ts.strftime('%Y%m%d_%H%M%S')}"
        self.frames = {cam: [] for cam in self.camera_names}
        self.timesteps = []

    def record_step(
        self, observation: dict[str, Any], action: dict[str, np.ndarray]
    ) -> None:
        """Append one rollout timestep to in-memory buffers.

        Args:
            observation (dict[str, Any]): Observation payload containing state
                and optional images.
            action (dict[str, np.ndarray]): Dictionary of action vector applied at this timestep.

        Returns:
            None: This method only updates in-memory buffers.
        """
        # TODO: Make sure all checks are present here
        self._check_step_data(observation, action)

        # Buffer frames for each configured camera (if available)
        for cam in self.camera_names:
            frame = self._get_camera_frame(observation["image_observation"], cam)
            if frame is not None:
                self.frames[cam].append(np.asarray(frame, dtype=np.uint8))

        # Buffer timestep data
        step_data = {"robot_state": {}, "action_dict": {}}
        for key in self.robot_profile.robot_state_keys:
            step_data["robot_state"][key] = np.asarray(
                observation["robot_state"][key], dtype=np.float32
            )
        step_data["action_dict"] = {
            "cartesian_position": action.get("cartesian_position", None),
            "cartesian_velocity": action.get("cartesian_velocity", None),
            "joint_position": action.get("joint_position", None),
            "joint_velocity": action.get("joint_velocity", None),
            "base_position": action.get("base_position", None),
            "base_velocity": action.get("base_velocity", None),
            "gripper_velocity": action.get("gripper_velocity", None),
            "gripper_position": action.get("gripper_position", None),
            "gripper_binary": action.get("gripper_binary", None),
        }
        self.timesteps.append(step_data)

    def _save_videos(
        self,
    ) -> tuple[str, dict[str, str]]:
        video_paths: dict[str, str] = {}
        self.session_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, frames in self.frames.items():
            video_path = self.session_dir / f"{self.save_fname}_{cam_name}.mp4"
            ImageSequenceClip(list(np.stack(frames)), fps=10).write_videofile(
                str(video_path),
                codec="libx264",
                logger=None,
            )
            video_paths[cam_name] = str(video_path.resolve())
        return video_paths

    def finish_rollout(self, instruction: str) -> None:
        # 1. Save videos under the recorder's per-session folder
        video_paths = self._save_videos()

        self.save(
            {
                "language_instruction": instruction,
                "metadata": {
                    "episode_id": self.save_fname,
                    "lab_id": self.lab_id,
                    "operator_name": self.operator_name,
                },
                "video_paths": video_paths,
            }
        )

    def save(self, data: dict[str, Any]) -> Path:
        """Persist the currently buffered episode to disk.

        Args:
            metadata (dict[str, Any]): Save metadata containing language and
                annotation fields.

        Returns:
            Path: Path to the written HDF5 episode file.

        Raises:
            ValueError: If no rollout steps were recorded.
        """
        if len(self.timesteps) == 0:
            raise ValueError("No steps recorded. Call record_step() first.")

        # # TODO: Check if this function is needed
        # normalized_metadata = self._normalize_metadata(md_in)

        # TODO: Check if this function is needed (most likely it is needed)
        data["video_paths"] = self._resolve_video_paths(
            output_dir=self.session_dir,
            provided_video_paths=data.get("video_paths", {}),
        )

        # Save HDF5 file to disk
        h5_filename = f"{self.save_fname}.h5"
        h5_path = self.session_dir / h5_filename
        self._save_h5(h5_path, data)

        return h5_path

    def _check_step_data(
        self, observation: dict[str, Any], action: dict[str, np.ndarray]
    ) -> None:
        """Validate observation and action inputs for a single rollout step.

        Args:
            observation (dict[str, Any]): Observation payload to validate.
            action (dict[str, np.ndarray]): Action dict to validate.

        Raises:
            ValueError: If required observation keys are missing, action is
                empty, action contains unrecognized keys, or all action values
                are None.
        """

        # Make sure the observation is a dictionary
        if not isinstance(observation, dict):
            raise ValueError(
                f"observation must be a dictionary, got {type(observation)}"
            )

        missing_obs = [k for k in REQUIRED_OBSERVATION_KEYS if k not in observation]
        if missing_obs:
            raise ValueError(
                f"observation is missing required keys: {missing_obs}. "
                f"Required: {REQUIRED_OBSERVATION_KEYS}. Please pass it in your record_step() call"
            )

        robot_state = observation["robot_state"]
        missing_robot_state_keys = [
            k for k in self.robot_profile.robot_state_keys if k not in robot_state
        ]
        if missing_robot_state_keys:
            raise ValueError(
                f"robot_state is missing required keys: {missing_robot_state_keys}. "
                f"Required: {self.robot_profile.robot_state_keys}. Please pass it in your record_step() call. Double check that the passed keys match the robot profile you initialized the recorder with."
            )

        image_observation = observation["image_observation"]
        missing_image_observation_keys = [
            k for k in self.robot_profile.camera_names if k not in image_observation
        ]
        if missing_image_observation_keys:
            raise ValueError(
                f"image_observation is missing required keys: {missing_image_observation_keys}. "
                f"Required: {self.robot_profile.camera_names} that you provided in the robot setup. Please pass it in your record_step() call. Double check that the passed keys match the robot profile you initialized the recorder with."
            )

        # Make sure the action is a dictionary
        if not isinstance(action, dict):
            raise ValueError(f"action must be a dictionary, got {type(action)}")

        # Make sure action has at least one key
        if not action:
            raise ValueError(
                f"action must not be empty. Valid keys: {VALID_ACTION_KEYS}. Please pass it in your record_step() call. Double check that the passed keys match the robot profile you initialized the recorder with."
            )

        # Make sure the action keys are valid
        invalid_action = set(action.keys()) - set(VALID_ACTION_KEYS)
        if invalid_action:
            raise ValueError(
                f"action contains unrecognized keys: {sorted(invalid_action)}. "
                f"Valid keys: {VALID_ACTION_KEYS}"
            )

        # Make sure the action keys agree between robot_profile and the action dict
        profile_action_keys = set(self.robot_profile.action_space)
        action_keys = set(action.keys())
        if action_keys != profile_action_keys:
            raise ValueError(
                f"action keys {action_keys} must match the robot profile action_space {profile_action_keys}"
            )

        # Make sure action values are not None
        if any(v is None for v in action.values()):
            raise ValueError(
                f"action contains None values for keys: {[k for k, v in action.items() if v is None]}. "
            )

        # TODO: It is currently only being called for cartesian_position, but we should also check for the other actions
        # If action is cartesian_position, verify the action shape
        for key in ("cartesian_position",):
            if key not in action:
                continue
            val = action.get(key)
            arr = np.asarray(val)
            # convert to quaternion if needed
            action[key] = (
                self.quat_conversion.convert_position(arr)
                if self.quat_conversion
                else arr
            )
            arr = np.asarray(action[key])
            if arr.shape not in ((7,), (14,)):
                raise ValueError(
                    f"action['{key}'] must have shape (7,) or (14,) — "
                    f"[x, y, z, qx, qy, qz, qw], got shape {arr.shape}"
                )
            quat_norm = np.linalg.norm(arr[3:7])
            if not np.isclose(quat_norm, 1.0, atol=1e-3):
                raise ValueError(
                    f"action['{key}'][3:7] must be a unit scalar-last quaternion (norm ≈ 1.0), "
                    f"got norm {quat_norm:.6f}"
                )
            # Heuristic: scalar (w) should not dominate vector part too heavily
            # (not guaranteed, but catches common mistakes)
            if abs(arr[6]) > 0.99 and np.linalg.norm(arr[3:6]) < 0.1:
                print(
                    f"Warning: Looks like action['{key}'][3:7] is in (w,x,y,z) order instead of (x,y,z,w)"
                )

        for key in ("cartesian_position",):
            if key not in robot_state:
                continue
            val = robot_state.get(key)
            arr = np.asarray(val)
            # convert to quaternion if needed
            robot_state[key] = (
                self.robot_state_quat_conversion.convert_position(arr)
                if self.robot_state_quat_conversion
                else arr
            )
            arr = np.asarray(robot_state[key])
            if arr.shape not in ((7,), (14,)):
                raise ValueError(
                    f"observation['{key}'] must have shape (7,) or (14,) — "
                    f"[x, y, z, qx, qy, qz, qw], got shape {arr.shape}"
                )
    # TODO: Polish this function!
    def _save_h5(self, path: Path, data: dict[str, Any]) -> None:
        """Write buffered rollout data and metadata into one HDF5 file.

        Args:
            path (Path): Target HDF5 file path.
            metadata (dict[str, Any]): Normalized metadata payload.

        Returns:
            None: This method only performs file I/O side effects.
        """
        str_dtype = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "w") as f:
            # 1. Save the metadata attributes
            f.attrs["schema"] = "oopsiedata_format_v1"
            f.attrs["episode_id"] = data["metadata"]["episode_id"]
            f.attrs.create(
                "robot_profile",
                robot_profile_to_json(self.robot_profile),
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            f.attrs["language_instruction"] = data.get("language_instruction", "")
            f.attrs["operator_name"] = data["metadata"]["operator_name"]
            f.attrs["lab_id"] = self.lab_id
            f.attrs["timestamp"] = self.timestamp

            # 2. Save the per-camera MP4 file paths (strings), not inlined frame tensors.
            observations_group = f.create_group("observations")
            video_paths_group = observations_group.create_group("video_paths")
            video_paths = data.get("video_paths", {})
            if not isinstance(video_paths, dict):
                video_paths = {}
            for cam in self.camera_names:
                raw_video_path = str(video_paths.get(cam, "")).strip()
                if not raw_video_path:
                    continue

                video_path_obj = Path(raw_video_path).expanduser()
                if video_path_obj.is_absolute():
                    rel_video_path = os.path.relpath(
                        video_path_obj.resolve(),
                        start=path.parent.resolve(),
                    )
                else:
                    rel_video_path = os.path.relpath(
                        (path.parent / video_path_obj).resolve(),
                        start=path.parent.resolve(),
                    )

                dataset = video_paths_group.create_dataset(
                    cam,
                    data=rel_video_path.replace(os.sep, "/"),
                    dtype=str_dtype,
                )

            # 3. Save the robot state data
            robot_states = observations_group.create_group("robot_states")
            for key in self.robot_profile.robot_state_keys:
                robot_states.create_dataset(
                    key,
                    data=np.stack(
                        [t["robot_state"][key] for t in self.timesteps], axis=0
                    ),
                    dtype=np.float64,
                )

            # 4. Save the action data
            action_group = f.create_group("actions")

            for action_key in self.timesteps[0]["action_dict"]:
                action_values = [t["action_dict"][action_key] for t in self.timesteps]
                if all(v is None for v in action_values):
                    action_group.create_dataset(
                        action_key, data=h5py.Empty(dtype=np.float64)
                    )
                else:
                    action_group.create_dataset(
                        action_key,
                        data=np.stack(action_values, axis=0),
                        dtype=np.float64,
                    )

            # eef_pos_values = [t["action_dict"]["cartesian_position"] for t in self.timesteps]
            # if all(v is None for v in eef_pos_values):
            #     action_group.create_dataset(
            #         "cartesian_position", data=h5py.Empty(dtype=np.float64)
            #     )
            # else:
            #     action_group.create_dataset(
            #         "cartesian_position",
            #         data=np.stack(eef_pos_values, axis=0),
            #         dtype=np.float64,
            #     )
            # eef_vel_values = [t["action_dict"]["cartesian_velocity"] for t in self.timesteps]
            # if all(v is None for v in eef_vel_values):
            #     action_group.create_dataset(
            #         "cartesian_velocity", data=h5py.Empty(dtype=np.float64)
            #     )
            # else:
            #     action_group.create_dataset(
            #         "cartesian_velocity",
            #         data=np.stack(eef_vel_values, axis=0),
            #         dtype=np.float64,
            #     )

            # joint_pos_values = [t["action_dict"]["joint_position"] for t in self.timesteps]
            # if all(v is None for v in joint_pos_values):
            #     action_group.create_dataset(
            #         "joint_position", data=h5py.Empty(dtype=np.float64)
            #     )
            # else:
            #     action_group.create_dataset(
            #         "joint_position",
            #         data=np.stack(joint_pos_values, axis=0),
            #         dtype=np.float64,
            #     )

            # joint_vel_values = [t["action_dict"]["joint_velocity"] for t in self.timesteps]
            # if all(v is None for v in joint_vel_values):
            #     action_group.create_dataset(
            #         "joint_velocity", data=h5py.Empty(dtype=np.float64)
            #     )
            # else:
            #     action_group.create_dataset(
            #         "joint_velocity",
            #         data=np.stack(joint_vel_values, axis=0),
            #         dtype=np.float64,
            #     )

    def _normalize_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        """Validate and normalize user-provided metadata.

        Args:
            metadata (dict[str, Any] | None): Raw metadata dictionary or
                ``None``.

        Returns:
            dict[str, Any]: Normalized metadata containing allowed top-level
                keys only.

        Raises:
            TypeError: If metadata or nested fields have invalid types.
            ValueError: If unsupported metadata keys are present.
        """
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise TypeError("metadata must be a dict")

        allowed_metadata_keys = {
            "language_instruction",
            "episode_annotations",
            "video_paths",
        }
        unknown_metadata_keys = set(metadata.keys()) - allowed_metadata_keys
        if unknown_metadata_keys:
            raise ValueError(
                "Unsupported metadata keys: "
                + ", ".join(sorted(unknown_metadata_keys))
                + ". Allowed keys: language_instruction, episode_annotations, video_paths"
            )

        episode_annotations = metadata.get("episode_annotations", {})
        if episode_annotations is None:
            episode_annotations = {}
        if not isinstance(episode_annotations, dict):
            raise TypeError(
                "metadata['episode_annotations'] must be a dict when provided"
            )

        allowed_annotation_keys = {
            "episode_id",
            "lab_id",
            "operator_name",
            "annotator_name",
            "success",
            "failure_annotation",
        }
        unknown_annotation_keys = (
            set(episode_annotations.keys()) - allowed_annotation_keys
        )
        if unknown_annotation_keys:
            raise ValueError(
                "Unsupported episode_annotations keys: "
                + ", ".join(sorted(unknown_annotation_keys))
                + ". Allowed keys: "
                + ", ".join(sorted(allowed_annotation_keys))
            )

        failure_annotation = episode_annotations.get("failure_annotation")
        if failure_annotation is not None and not isinstance(
            failure_annotation, (dict, str)
        ):
            raise TypeError(
                "metadata['episode_annotations']['failure_annotation'] must be a dict or str"
            )

        video_paths = metadata.get("video_paths", {})
        if video_paths is None:
            video_paths = {}
        if not isinstance(video_paths, dict):
            raise TypeError("metadata['video_paths'] must be a dict when provided")

        return {
            "language_instruction": str(metadata.get("language_instruction", "")),
            "episode_annotations": dict(episode_annotations),
            "video_paths": dict(video_paths),
        }

    def _resolve_video_paths(
        self,
        output_dir: Path,
        provided_video_paths: Any,
    ) -> dict[str, str]:
        """Resolve per-camera MP4 paths, writing videos when needed.

        Args:
            output_dir (Path): Directory used as the base for relative path
                storage.
            provided_video_paths (Any): Optional external mapping from camera to
                MP4 path.

        Returns:
            dict[str, str]: Mapping from camera name to relative MP4 path.
        """
        paths: dict[str, str] = {}
        provided = (
            provided_video_paths if isinstance(provided_video_paths, dict) else {}
        )

        for cam in self.camera_names:
            provided_path = str(provided.get(cam, ""))
            if provided_path:
                abs_path = Path(provided_path).expanduser().resolve()
                if abs_path.suffix.lower() != ".mp4":
                    abs_path = abs_path.with_suffix(".mp4")
                paths[cam] = os.path.relpath(abs_path, start=output_dir)
                continue

            frames = self.frames.get(cam, [])
            if len(frames) == 0:
                continue

            video_path = output_dir / f"{self.save_fname}_{cam}.mp4"
            write_mp4(video_path=video_path, frames=np.asarray(frames), fps=10.0)
            paths[cam] = os.path.relpath(video_path.resolve(), start=output_dir)

        return paths

    def _get_camera_frame(
        self, observation: dict[str, Any], cam_name: str
    ) -> np.ndarray | None:
        """Extract a camera frame from supported observation key patterns.

        Args:
            observation (dict[str, Any]): Observation dictionary for one
                timestep.
            cam_name (str): Canonical camera name to resolve.

        Returns:
            np.ndarray | None: Camera frame array when available, otherwise
                ``None``.
        """
        candidates = (cam_name, f"image_{cam_name}", f"{cam_name}_image")
        for key in candidates:
            if key in observation:
                return np.asarray(observation[key])
        return None

    def patch_h5_failure_annotation(
        h5_path: Path,
        annotation: dict[str, Any],
    ) -> None:
        """Patch an existing episode HDF5 with a human annotation.

        This is used when the episode is saved immediately after rollout and the
        human annotation arrives later.
        """
        if not h5_path.exists():
            raise FileNotFoundError(str(h5_path))

        success: float | None = None
        bs = str(annotation.get("binary_success", "")).strip().lower()
        if bs == "success":
            success = 1.0
        elif bs == "failure":
            success = 0.0

        with h5py.File(h5_path, "r+") as f:
            episode_annotations = f.require_group("episode_annotations")
            annotation_group = episode_annotations.require_group(
                annotation["annotator"]
            )
            annotation_group.attrs["schema"] = annotation.get("schema", "oopsie_failure_taxonomy_v1")
            annotation_group.attrs["source"] = "human"
            annotation_group.attrs["timestamp"] = annotation["annotated_at"]
            annotation_group.attrs["success"] = success
            annotation_group.attrs["failure_description"] = annotation[
                "failure_description"
            ]
            annotation_group.attrs["taxonomy_schema"] = "oopsiedata_taxonomy_schema_v1"
            failure_taxonomy = {
                "failure_category": annotation["failure_category"],
                "severity": annotation["severity"],
            }
            annotation_group.attrs["taxonomy"] = json.dumps(
                failure_taxonomy, ensure_ascii=False
            )
            annotation_group.attrs["additional_notes"] = annotation["additional_notes"]

            # annotation_group.attrs["failure_annotation"] = json.dumps(
            #     dict(annotation), ensure_ascii=False
            # )

    @property
    def num_steps(self) -> int:
        """Return the number of recorded timesteps.

        Returns:
            int: Number of timesteps currently buffered.
        """
        return len(self.timesteps)


class MinimalEpisodeRecorder(EpisodeRecorder):
    """Episode recorder variant that persists data without metadata input."""

    def save(self, metadata: dict[str, Any] | None = None) -> Path:
        """Persist the buffered episode with empty/default metadata.

        Args:
            metadata (dict[str, Any] | None): Unused compatibility argument.

        Returns:
            Path: Path to the written HDF5 episode file.

        Raises:
            ValueError: If no rollout steps were recorded.
        """
        if len(self.timesteps) == 0:
            raise ValueError("No steps recorded. Call record_step() first.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        h5_filename = f"{self.session_name}.h5"
        h5_path = self.output_dir / h5_filename
        self._save_h5(h5_path, metadata={})

        return h5_path


class InteractiveAnnotator:
    """Command-line helper that prompts users for episode annotations."""

    def prompt(self, language_instruction: str = "") -> dict[str, Any]:
        """Prompt for annotation fields and return recorder-ready metadata.

        Args:
            language_instruction (str): Optional pre-filled instruction string.

        Returns:
            dict[str, Any]: Metadata dictionary compatible with
                ``EpisodeRecorder.save``.
        """
        print("\n" + "=" * 50)
        print("Episode Annotation")
        print("=" * 50)

        # Language instruction
        if language_instruction:
            print(f"\nInstruction: {language_instruction}")
            edit = input("Edit instruction? [y/N]: ").strip().lower()
            if edit == "y":
                language_instruction = input("New instruction: ").strip()
        else:
            language_instruction = input("Language instruction: ").strip()

        # Success
        while True:
            success_input = input("\nSuccess score (0.0-1.0): ").strip()
            try:
                success = float(success_input)
                if 0.0 <= success <= 1.0:
                    break
                print("Please enter a value between 0.0 and 1.0")
            except ValueError:
                print("Please enter a valid number")

        # Questionnaire reply payload
        print(
            "\nQuestionnaire response (describe what happened, press Enter twice to finish):"
        )
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        question = "\n".join(lines)

        print("=" * 50 + "\n")

        return {
            "language_instruction": language_instruction,
            "episode_annotations": {
                "success": success,
                "failure_annotation": {
                    "question": question,
                },
            },
        }
