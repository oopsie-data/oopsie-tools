"""Load robot / lab profiles for annotation and episode recording."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import yaml

from oopsie_tools.utils.rotation_utils import RotOption

ACTION_SPACE_SET_1 = {
    "joint_position",
    "joint_velocity",
    "cartesian_position",
    "cartesian_velocity",
}

ACTION_SPACE_SET_2 = {
    "gripper_position", 
    "gripper_velocity", 
    "gripper_binary",
}

ACTION_SPACE_SET_3 = {
    "base_velocity", 
    "base_position",
}

REQUIRED_KEYS = frozenset(
    {
        "policy_name",
        "robot_name",
        "gripper_name",
        "control_freq",
        "is_biarm",
        "uses_mobile_base",
        "camera_names",
        "robot_state_keys",
        "action_space",
    }
)

REQUIRED_ROBOT_STATE_KEYS = frozenset(
    {
        "joint_position",
        "gripper_position",
    }
)


@dataclasses.dataclass(frozen=True)
class RobotProfile:
    """Robot / dataset identity and recording semantics (paths, joints, cameras).

    Options specific to :class:`WebRolloutAnnotator` (browser server port, blocking
    until annotation, resuming a session directory) are *not* part of this profile;
    pass those separately when constructing the annotator, e.g. via
    :meth:`WebRolloutAnnotator.from_robot_profile`.
    """

    policy_name: str
    robot_name: str
    is_biarm: bool
    uses_mobile_base: bool
    gripper_name: str
    control_freq: int
    camera_names: list[str]
    robot_state_keys: list[str]
    robot_state_joint_names: list[str]
    action_space: list[str]
    action_joint_names: list[str] | None = None
    orientation_representation: str | None = None
    robot_state_orientation_representation: str | None = None
    controller: str | None = None
    gains: dict[str, Any] | None = None
    intrinsic_calibration_matrix: dict[str, Any] | None = None
    extrinsic_calibration_matrix: dict[str, Any] | None = None

    def get_rot_option(self) -> RotOption | None:
        if self.orientation_representation is None:
            return None
        return RotOption.from_string(self.orientation_representation)

    def get_robot_state_rot_option(self) -> RotOption | None:
        if self.robot_state_orientation_representation is None:
            return None
        return RotOption.from_string(self.robot_state_orientation_representation)


def robot_profile_to_json(profile: RobotProfile) -> str:
    """Serialize ``RobotProfile`` to a JSON string (for HDF5 file attributes)."""
    return json.dumps(dataclasses.asdict(profile), ensure_ascii=False)


def robot_profile_config_dir() -> Path:
    """Directory containing bundled ``*.yaml`` robot profiles."""
    return Path(__file__).resolve().parent.parent.parent / "configs" / "robot_profiles"


def default_robot_profile_path() -> Path:
    return robot_profile_config_dir() / "default_robot_profile.yaml"


def openpi_example_robot_profile_path() -> Path:
    return robot_profile_config_dir() / "openpi_example_robot_profile.yaml"


def act_plus_plus_robot_profile_path() -> Path:
    return robot_profile_config_dir() / "act_plus_plus_robot_profile.yaml"


def mock_robot_profile_path() -> Path:
    return robot_profile_config_dir() / "mock_robot_profile.yaml"


def load_robot_profile(path: Path | str) -> RobotProfile:
    """Parse a robot profile YAML file into ``Robotprofile``."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Robot profile file not found: {p}")
    raw = yaml.safe_load(p.read_text())
    return robot_profile_from_raw(raw)


def robot_profile_from_json(payload: str) -> RobotProfile:
    """Parse a robot profile JSON string into ``RobotProfile``."""
    try:
        raw = json.loads(payload)
    except Exception as e:
        raise ValueError(f"Invalid robot_profile JSON: {e}")
    return robot_profile_from_raw(raw)


def is_valid_action_space(action_space):
    s = set(action_space)

    arm_count = len(s & ACTION_SPACE_SET_1)
    gripper_count = len(s & ACTION_SPACE_SET_2)
    base_count = len(s & ACTION_SPACE_SET_3)

    return (
        arm_count >= 1 and
        gripper_count >= 1 and
        base_count <= 1 and
        len(s) == arm_count + gripper_count + base_count  # no extras
    )

def robot_profile_from_raw(raw: Any) -> RobotProfile:
    """Validate and parse raw mapping data into ``RobotProfile``."""
    if not isinstance(raw, dict):
        raise ValueError(f"Robot profile must be a mapping, got {type(raw).__name__}")

    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(f"Robot profile missing keys: {missing}")

    missing_robot_state_keys = [
        k for k in REQUIRED_ROBOT_STATE_KEYS if k not in raw.get("robot_state_keys")
    ]
    if missing_robot_state_keys:
        raise ValueError(
            f"Robot profile missing robot state keys: {missing_robot_state_keys}"
        )

    action_space = list(raw["action_space"])
    if not is_valid_action_space(action_space):
        raise ValueError(
            f"Invalid action_space: {action_space!r}. "
            "Expected exactly 1 arm action from "
            f"{sorted(ACTION_SPACE_SET_1)}, "
            "exactly 1 gripper action from "
            f"{sorted(ACTION_SPACE_SET_2)}, "
            "and optionally 1 base action from "
            f"{sorted(ACTION_SPACE_SET_3)}."
        )

    action_joint_names = _optional_str_list(raw.get("action_joint_names"))
    if (
        any(k in {"joint_position", "joint_velocity"} for k in action_space)
        and not action_joint_names
    ):
        raise ValueError(
            "action_joint_names is required for joint_position and joint_velocity "
            "action spaces"
        )

    # check validity of action space
    valid_gripper_set = {"gripper_position", "gripper_velocity", "gripper_binary"}
    valid_base_set = {"base_velocity", "base_position"}
    uses_mobile_base = bool(raw.get("uses_mobile_base", False))

    if set(action_space).isdisjoint(valid_gripper_set):
        raise ValueError(
            f"Invalid action_space {action_space!r}: must include at least one of "
            f"{valid_gripper_set}"
        )
    if uses_mobile_base and set(action_space).isdisjoint(valid_base_set):
        raise ValueError(
            f"Invalid action_space {action_space!r} for mobile base: must include at least one of "
            f"{valid_base_set}"
        )

    return RobotProfile(
        policy_name=raw["policy_name"],
        robot_name=raw["robot_name"],
        is_biarm=raw.get("is_biarm", False),  # default to False if not specified
        uses_mobile_base=raw.get("uses_mobile_base", False),  # default to False if not specified
        gripper_name=raw["gripper_name"],
        control_freq=raw["control_freq"],
        camera_names=list(raw["camera_names"]),
        # Observation Related
        robot_state_keys=list(raw["robot_state_keys"]),
        robot_state_joint_names=list(raw["robot_state_joint_names"]),
        # Action Related
        action_space=action_space,
        action_joint_names=action_joint_names,
        orientation_representation=raw.get("orientation_representation", None),
        # Optional Keys
        robot_state_orientation_representation=raw.get("robot_state_orientation_representation", None),
        controller=raw.get("controller"),
        gains=raw.get("gains"),
        intrinsic_calibration_matrix=raw.get("intrinsic_calibration_matrix"),
        extrinsic_calibration_matrix=raw.get("extrinsic_calibration_matrix"),
    )


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    raise ValueError(f"Expected a list or null, got {type(value).__name__}")
