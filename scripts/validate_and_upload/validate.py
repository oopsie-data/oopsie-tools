"""
Validate HDF5 episodes against the ``robotic_failure_upload_data_format_v1`` schema.

Usage:
    python validate.py --samples_dir /path/to/formatted_data               # all *.h5 in folder
    python validate.py --samples_dir /path/to/formatted_data --episode_id 000001  # single episode
"""

import json
import os
import h5py
import glob
import cv2
import argparse
import sys
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
from oopsie_tools.utils.robot_profile import robot_profile_from_json


# ── Gap analysis: known mapping from ACT/ALOHA format to this dataset format ──
# Status legend:
#   direct      — copy as-is (field name/structure matches)
#   transform   — source field exists but needs reshaping / extraction
#   encode      — source has raw frames; must encode to mp4 and store path
#   placeholder — no source field; use zeros/constant until FK is available
#   manual      — requires human annotation
_ALOHA_FIELD_MAP = [
    {
        "target": "action_dict/cartesian_position",
        "description": "EEF cartesian pose (T, 6) — [x,y,z,rx,ry,rz]",
        "aloha_source": None,
        "status": "placeholder",
        "notes": "Forward kinematics on action joint angles not available — use np.zeros((T,6))",
    },
    {
        "target": "action_dict/cartesian_velocity",
        "description": "EEF cartesian velocity (T, 6)",
        "aloha_source": None,
        "status": "placeholder",
        "notes": "Forward kinematics not available — use np.zeros((T,6))",
    },
    {
        "target": "action_dict/gripper_position",
        "description": "Gripper position (T,)",
        "aloha_source": "action[:, 6]",
        "status": "transform",
        "notes": "Last joint of left arm (index 6); right arm at index 13",
    },
    {
        "target": "action_dict/gripper_velocity",
        "description": "Gripper velocity (T,)",
        "aloha_source": "action[:, 6]",
        "status": "transform",
        "notes": "np.gradient(action[:, 6])",
    },
    {
        "target": "image_observations/exterior_image_1_left",
        "description": "Path to exterior camera 1 mp4 (scalar string)",
        "aloha_source": "observations/images/cam_high",
        "status": "encode",
        "notes": "Encode (T, H, W, 3) uint8 frames → mp4; resize to ≤1080px (source is 1280×720)",
    },
    {
        "target": "image_observations/exterior_image_2_left",
        "description": "Path to exterior camera 2 mp4 (scalar string)",
        "aloha_source": "observations/images/cam_right_wrist",
        "status": "encode",
        "notes": "Encode (T, H, W, 3) uint8 frames → mp4; resize to ≤1080px (source is 1280×720)",
    },
    {
        "target": "image_observations/wrist_image_left",
        "description": "Path to wrist camera mp4 (scalar string)",
        "aloha_source": "observations/images/cam_left_wrist",
        "status": "encode",
        "notes": "Encode (T, H, W, 3) uint8 frames → mp4; resize to ≤1080px (source is 1280×720)",
    },
    {
        "target": "episode_annotations/success",
        "description": "Episode success label (scalar float: 1.0=success, 0.0=failure)",
        "aloha_source": None,
        "status": "manual",
        "notes": "Watch the episode and annotate; cannot be inferred from data alone",
    },
    {
        "target": "episode_annotations/episode_id",
        "description": 'Episode ID string (e.g. "000001")',
        "aloha_source": None,
        "status": "manual",
        "notes": "Derive from output filename: zero-padded 6-digit integer",
    },
    {
        "target": "observation/cartesian_position",
        "description": "Observed EEF cartesian pose (T, 6)",
        "aloha_source": None,
        "status": "placeholder",
        "notes": "Forward kinematics on observations/qpos not available — use np.zeros((T,6))",
    },
    {
        "target": "observation/gripper_position",
        "description": "Observed gripper position (T,)",
        "aloha_source": "observations/qpos[:, 6]",
        "status": "transform",
        "notes": "Last joint of left arm (index 6); right arm at index 13",
    },
    {
        "target": "observation/joint_position",
        "description": "All joint positions (T, 14)",
        "aloha_source": "observations/qpos",
        "status": "direct",
        "notes": "Direct copy — rename only",
    },
    {
        "target": "language_instruction",
        "description": "Task description (scalar string)",
        "aloha_source": None,
        "status": "manual",
        "notes": 'Provide a natural language instruction, e.g. "stack the tote on top of the other totes"',
    },
]

_STATUS_ICON = {
    "direct": "✓",
    "transform": "~",
    "encode": "⊞",
    "placeholder": "○",
    "manual": "✗",
}

_STATUS_LABEL = {
    "direct": "direct copy",
    "transform": "transform",
    "encode": "encode→mp4",
    "placeholder": "placeholder",
    "manual": "manual",
}


def gap_analysis(source_h5_path: str) -> None:
    """
    Print a gap analysis comparing a source HDF5 file against the required
    robotic failure dataset schema.

    Args:
        source_h5_path: Path to a source HDF5 file (e.g. ACT/ALOHA format).
    """
    print(
        f"\nGAP ANALYSIS: {os.path.basename(source_h5_path)} → robotic failure dataset format"
    )
    print("=" * 80)

    # Collect all keys in source file
    source_keys: dict[str, tuple] = {}

    def _collect(name, obj):
        if isinstance(obj, h5py.Dataset):
            source_keys[name] = obj.shape

    try:
        with h5py.File(source_h5_path, "r") as f:
            f.visititems(_collect)
    except Exception as e:
        print(f"ERROR: Could not open source file: {e}")
        return

    # Print source structure
    print(f"\nSource file fields ({len(source_keys)} datasets):")
    for k, shape in sorted(source_keys.items()):
        print(f"  {k:<50} {str(shape)}")

    # Image size warning
    img_keys = [k for k in source_keys if "image" in k or "cam" in k]
    needs_resize = False
    for k in img_keys:
        shape = source_keys[k]
        if len(shape) >= 3 and (
            shape[-2] > MAX_IMAGE_SIZE or shape[-3] > MAX_IMAGE_SIZE
        ):
            needs_resize = True
            break

    # Print mapping table
    print("\nRequired target fields vs source:")
    W_TARGET = 46
    W_SOURCE = 32
    W_STATUS = 13
    sep = "─" * (W_TARGET + W_SOURCE + W_STATUS + 6)
    print(sep)
    print(
        f" {'':2} {'Target field':<{W_TARGET}} {'Source field':<{W_SOURCE}} {'Action':<{W_STATUS}}"
    )
    print(sep)

    counts = {k: 0 for k in _STATUS_ICON}
    for entry in _ALOHA_FIELD_MAP:
        icon = _STATUS_ICON[entry["status"]]
        label = _STATUS_LABEL[entry["status"]]
        source = entry["aloha_source"] or "—"
        print(f" {icon}  {entry['target']:<{W_TARGET}} {source:<{W_SOURCE}} {label}")
        counts[entry["status"]] += 1

    print(sep)

    # Summary
    print("\nSummary:")
    print(f"  {_STATUS_ICON['direct']}  direct copy   : {counts['direct']}")
    print(f"  {_STATUS_ICON['transform']}  transform     : {counts['transform']}")
    print(f"  {_STATUS_ICON['encode']}  encode→mp4    : {counts['encode']}")
    print(
        f"  {_STATUS_ICON['placeholder']}  placeholder   : {counts['placeholder']}  (zeros until FK available)"
    )
    print(
        f"  {_STATUS_ICON['manual']}  manual        : {counts['manual']}  (requires human annotation)"
    )

    if needs_resize:
        print(
            f"\n  ⚠  Image resize needed: source frames exceed MAX_IMAGE_SIZE={MAX_IMAGE_SIZE}px."
        )
        print(f"     Detected large dims in: {', '.join(img_keys)}")
        print(
            f"     Resize to fit within {MAX_IMAGE_SIZE}×{MAX_IMAGE_SIZE} before encoding."
        )

    print("\nTo convert, run:")
    print(f"  python convert.py --source {source_h5_path} --samples_dir <output_dir>\n")


MAX_IMAGE_SIZE = 1280
MIN_IMAGE_SIZE = 180
MIN_TRAJECTORY_LENGTH = 2
MAX_TRAJECTORY_LENGTH = 300

ROBOTIC_FAILURE_UPLOAD_SCHEMA_V1 = "robotic_failure_upload_data_format_v1"
OOPSIE_DATA_SCHEMA_V1 = "oopsiedata_format_v1"

_OOPSIE_REQUIRED_ROOT_ATTRS = (
    "schema",
    "episode_id",
    "language_instruction",
    "lab_id",
    "operator_name",
    "robot_profile",
)


@dataclass
class DataInstance:
    base_path: Optional[str] = None
    episode_id: Optional[str] = None
    # If set, open this HDF5 path directly (session layout: arbitrary *.h5 names).
    h5_path: Optional[str] = None
    h5_file: Optional[h5py.File] = None
    observations: Optional[dict] = None
    actions: Optional[dict] = None
    left1_video_path: Optional[str] = None
    left2_video_path: Optional[str] = None
    wrist_video_path: Optional[str] = None
    """(camera_key, absolute_mp4_path) from image_observations; used for all schema tests."""
    video_paths_list: Optional[List[Tuple[str, str]]] = None
    """``upload_v1`` (session recorder) or ``legacy`` (ALOHA-style layout)."""
    h5_schema: Optional[str] = None
    annotations: Optional[dict] = None
    language_instruction: Optional[str] = None
    failure_category: Optional[str] = None
    failure_description: Optional[str] = None
    would_retry: Optional[bool] = None
    trajectory_length: Optional[int] = None
    control_freq: Optional[float] = None


_REGISTERED_TESTS: List[Callable[[DataInstance], None]] = []


def register_test(
    name: str,
) -> Callable[[Callable[[DataInstance], None]], Callable[[DataInstance], None]]:
    """
    Register a validation test for the dataset.

    Args:
        name: Human-readable name for the test

    Returns:
        Decorator that registers the test function
    """

    def decorator(
        func: Callable[[DataInstance], None],
    ) -> Callable[[DataInstance], None]:
        func.test_name = name
        _REGISTERED_TESTS.append(func)
        return func

    return decorator


def _run_registered_tests(
    context: DataInstance, tests: Optional[List[Callable[[DataInstance], None]]] = None
) -> None:
    selected_tests = tests or _REGISTERED_TESTS
    total_tests = len(selected_tests)

    for i, test in enumerate(selected_tests, 1):
        test_name = getattr(test, "test_name", test.__name__)

        # Calculate progress bar
        progress = (i - 1) / total_tests
        bar_length = 20
        filled = int(bar_length * progress)
        bar = "█" * filled + "░" * (bar_length - filled)

        # Print progress bar with current test
        sys.stdout.write(
            f"\rRunning tests: [{bar}] {i}/{total_tests} ({int(progress * 100)}%) - {test_name}"
        )
        sys.stdout.flush()

        # Run the test
        test(context)

    # Print final progress bar
    bar = "█" * 20
    sys.stdout.write(f"\rRunning tests: [{bar}] {total_tests}/{total_tests} (100%)\n")
    sys.stdout.flush()


def _resolve_h5_path(context: DataInstance) -> str:
    """Path to the episode HDF5: explicit h5_path or legacy {episode_id}_trajectory.h5."""
    if context.h5_path:
        return os.path.abspath(os.path.normpath(context.h5_path))
    assert context.base_path is not None and context.episode_id is not None
    return os.path.join(context.base_path, f"{context.episode_id}_trajectory.h5")


def _decode_h5_scalar(value: Any) -> str:
    """Normalize HDF5 attr / scalar dataset values to str."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        return _decode_h5_scalar(value.item())
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_h5_scalar(value.item())
    return str(value)


def _read_scalar_string_dataset(ds: h5py.Dataset) -> str:
    raw = ds[()]
    return _decode_h5_scalar(raw).strip()


def _is_oopsie_v1(f: h5py.File) -> bool:
    """True for episodes written by EpisodeRecorder (oopsiedata_format_v1)."""
    sch = f.attrs.get("schema")
    return sch is not None and _decode_h5_scalar(sch) == OOPSIE_DATA_SCHEMA_V1


def _is_robotic_failure_upload_v1(f: h5py.File) -> bool:
    """True for the legacy upload_v1 ALOHA-style schema."""
    sch = f.attrs.get("schema")
    if sch is not None and _decode_h5_scalar(sch) == ROBOTIC_FAILURE_UPLOAD_SCHEMA_V1:
        return True
    # Heuristic: recorder stores instruction on the file, not as a top-level dataset.
    if "language_instruction" in f.attrs and "language_instruction" not in f:
        return True
    return False


def _episode_annotations_as_dict(ea: h5py.Group) -> dict:
    """Merge group attrs and datasets (upload v1 uses attrs for fixed metadata)."""
    out: dict = {}
    for k in ea.attrs:
        out[k] = ea.attrs[k]
    for k in ea.keys():
        out[k] = ea[k]
    return out


def _video_min_duration_seconds(context: DataInstance) -> float:
    """Session rollouts are short (few seconds); relax vs. legacy constant."""
    if context.h5_schema == "upload_v1" and context.trajectory_length:
        return max(
            0.15, min(float(MIN_TRAJECTORY_LENGTH), context.trajectory_length / 12.0)
        )
    return float(MIN_TRAJECTORY_LENGTH)


def _collect_camera_video_entries(
    h5_file: h5py.File, h5_dir: str
) -> Tuple[List[Tuple[str, str]], List[str]]:
    """Collect per-camera mp4 paths from recorder-preferred or fallback layouts."""
    video_entries: List[Tuple[str, str]] = []
    cam_keys: List[str] = []

    recorder_group = (
        h5_file["observations"]["video_paths"]
        if "observations" in h5_file and "video_paths" in h5_file["observations"]
        else None
    )

    if recorder_group is not None and len(recorder_group.keys()) > 0:
        cam_keys = sorted(list(recorder_group.keys()))
        for ck in cam_keys:
            ds = recorder_group[ck]
            assert isinstance(ds, h5py.Dataset), (
                f"observations/video_paths/{ck} must be a dataset"
            )
            rel = _read_scalar_string_dataset(ds)
            assert rel, f"Empty video path for camera {ck}"
            abs_p = os.path.normpath(os.path.join(h5_dir, rel))
            video_entries.append((str(ck), abs_p))
        return video_entries, cam_keys

    fallback_group = h5_file["image_observations"] if "image_observations" in h5_file else None
    assert fallback_group is not None and len(fallback_group.keys()) > 0, (
        "No camera video mapping found. Expected observations/video_paths or image_observations"
    )
    cam_keys = sorted(list(fallback_group.keys()))
    for ck in cam_keys:
        ds = fallback_group[ck]
        assert isinstance(ds, h5py.Dataset), (
            f"image_observations/{ck} must be a dataset"
        )
        rel = _read_scalar_string_dataset(ds)
        assert rel, f"Empty video path for camera {ck}"
        abs_p = os.path.normpath(os.path.join(h5_dir, rel))
        video_entries.append((str(ck), abs_p))
    return video_entries, cam_keys


@register_test("readable")
def _validate_h5_file_readable(context: DataInstance) -> bool:
    """
    Validate that the h5 file exists and is readable.

    Args:
        context: DataInstance containing the h5 file path

    Returns:
        True if file is readable

    Raises:
        AssertionError: If file does not exist or is not readable
    """
    h5_path = _resolve_h5_path(context)
    assert os.path.exists(h5_path), f"H5 file does not exist: {h5_path}"
    assert os.path.isfile(h5_path), f"H5 path is not a file: {h5_path}"

    try:
        f = h5py.File(h5_path, "r")
        context.h5_file = f
    except Exception as e:
        raise AssertionError(f"H5 file is not readable: {h5_path}. Error: {e}")

    return True


@register_test("required_keys")
def _validate_required_keys(context: DataInstance) -> None:
    """
    Validate that all required keys are present in the h5 file structure.

    Supports:

    - **upload_v1**: ``robotic_failure_upload_data_format_v1`` from
      ``episode_recorder`` (language_instruction file attr, dynamic camera keys,
      episode metadata in ``episode_annotations`` attrs).
    - **legacy**: ALOHA-style top-level ``language_instruction`` dataset and
      fixed ``exterior_image_*`` / ``wrist_image_left`` keys.
    """
    h5_file = context.h5_file
    assert h5_file is not None
    base_path = context.base_path
    assert base_path is not None
    h5_dir = os.path.abspath(os.path.normpath(base_path))

    if _is_oopsie_v1(h5_file):
        context.h5_schema = "oopsie_v1"

        for attr in _OOPSIE_REQUIRED_ROOT_ATTRS:
            assert attr in h5_file.attrs, f"Missing root attr: {attr}"

        try:
            profile = robot_profile_from_json(
                _decode_h5_scalar(h5_file.attrs["robot_profile"])
            )
        except ValueError as e:
            raise AssertionError(str(e))

        assert profile.camera_names, "robot_profile camera_names must be non-empty"
        assert profile.control_freq > 0, "robot_profile control_freq must be > 0"
        context.control_freq = float(profile.control_freq)

        assert "actions" in h5_file, "Missing group: actions"
        assert "observations" in h5_file and "robot_states" in h5_file["observations"], (
            "Missing group: observations/robot_states"
        )

        rs = h5_file["observations/robot_states"]
        for key in profile.robot_state_keys:
            assert key in rs, f"Missing observations/robot_states key: {key}"

        action_group = h5_file["actions"]
        for key in profile.action_space:
            assert key in action_group, f"Missing actions key from profile.action_space: {key}"

        video_entries, cam_keys = _collect_camera_video_entries(h5_file, h5_dir)
        missing_cams = sorted(set(profile.camera_names) - set(cam_keys))
        assert not missing_cams, (
            f"Missing camera video paths for profile cameras: {missing_cams}"
        )

        jp = rs.get("joint_position")
        if jp is not None and len(jp.shape) >= 2:
            assert len(profile.robot_state_joint_names) == int(jp.shape[-1]), (
                "robot_state_joint_names length does not match observations/robot_states/joint_position dof"
            )

        action_joint_names = profile.action_joint_names or []
        for joint_action_key in ("joint_position", "joint_velocity"):
            ds = action_group.get(joint_action_key)
            if ds is None or ds.shape is None or len(ds.shape) < 2:
                continue
            if action_joint_names:
                assert len(action_joint_names) == int(ds.shape[-1]), (
                    f"action_joint_names length does not match actions/{joint_action_key} dof"
                )

        context.video_paths_list = video_entries

        context.observations = dict(h5_file["observations/robot_states"])
        context.actions = dict(action_group)
        context.language_instruction = _decode_h5_scalar(h5_file.attrs["language_instruction"])
        return

    if _is_robotic_failure_upload_v1(h5_file):
        context.h5_schema = "upload_v1"

        for key in (
            "action_dict",
            "image_observations",
            "episode_annotations",
            "observation",
        ):
            assert key in h5_file, f"Missing top-level key: {key}"

        assert "language_instruction" in h5_file.attrs, (
            "Missing file attr language_instruction"
        )
        context.language_instruction = _decode_h5_scalar(
            h5_file.attrs["language_instruction"]
        )

        required_action = (
            "cartesian_position",
            "cartesian_velocity",
            "gripper_position",
            "gripper_velocity",
        )
        ad = h5_file["action_dict"]
        for key in required_action:
            assert key in ad, f"Missing action_dict key: {key}"

        img = h5_file["image_observations"]
        cam_keys = list(img.keys())
        assert cam_keys, "image_observations must contain at least one camera dataset"

        video_entries: List[Tuple[str, str]] = []
        for ck in sorted(cam_keys):
            ds = img[ck]
            assert isinstance(ds, h5py.Dataset), (
                f"image_observations/{ck} must be a dataset"
            )
            rel = _read_scalar_string_dataset(ds)
            assert rel, f"Empty video path for camera {ck}"
            abs_p = os.path.normpath(os.path.join(h5_dir, rel))
            video_entries.append((str(ck), abs_p))
        context.video_paths_list = video_entries

        ea = h5_file["episode_annotations"]
        required_ea_attrs = (
            "episode_id",
            "lab_id",
            "operator_name",
            "policy_id",
            "robot_id",
            "control_freq",
            "success",
        )
        for ak in required_ea_attrs:
            assert ak in ea.attrs, f"Missing episode_annotations attr: {ak}"

        og = h5_file["observation"]
        for ok in ("cartesian_position", "gripper_position", "joint_position"):
            assert ok in og, f"Missing observation key: {ok}"

        context.observations = dict(h5_file["observation"])
        context.actions = dict(h5_file["action_dict"])
        context.annotations = _episode_annotations_as_dict(ea)
        return

    # ── Legacy ALOHA-style layout ───────────────────────────────────────────
    context.h5_schema = "legacy"

    required_top_level = [
        "action_dict",
        "image_observations",
        "episode_annotations",
        "observation",
        "language_instruction",
    ]
    required_action_dict = [
        "cartesian_position",
        "cartesian_velocity",
        "joint_position",
        "joint_velocity",
    ]
    required_image_obs = [
        "exterior_image_1_left",
        "exterior_image_2_left",
        "wrist_image_left",
    ]
    required_episode_annotations = [
        "success",
        "episode_id",
        "lab_id",
        "policy_id",
        "robot_id",
        "control_freq",
    ]
    required_observation = [
        "cartesian_position",
        "gripper_position",
        "joint_position",
    ]

    for key in required_top_level:
        assert key in h5_file, f"Missing top-level key: {key}"

    assert any(key in h5_file["action_dict"] for key in required_action_dict), (
        f"Missing all action_dict keys. At least one of {required_action_dict} must be present"
    )

    for key in required_image_obs:
        assert key in h5_file["image_observations"], (
            f"Missing image_observations key: {key}"
        )

    for key in required_episode_annotations:
        assert key in h5_file["episode_annotations"], (
            f"Missing episode_annotations key: {key}"
        )

    for key in required_observation:
        assert key in h5_file["observation"], f"Missing observation key: {key}"

    assert h5_file["action_dict"] is not None, "action_dict is None"
    assert h5_file["image_observations"] is not None, "image_observations is None"
    assert h5_file["episode_annotations"] is not None, "episode_annotations is None"
    assert h5_file["observation"] is not None, "observation is None"
    assert h5_file["language_instruction"][()] is not None, (
        "language_instruction is None"
    )
    assert h5_file["image_observations"]["exterior_image_1_left"][()] is not None, (
        "exterior_image_1_left is None"
    )
    assert h5_file["image_observations"]["exterior_image_2_left"][()] is not None, (
        "exterior_image_2_left is None"
    )
    assert h5_file["image_observations"]["wrist_image_left"][()] is not None, (
        "wrist_image_left is None"
    )

    context.observations = dict(h5_file["observation"])
    context.actions = dict(h5_file["action_dict"])
    context.annotations = dict(h5_file["episode_annotations"])
    li = h5_file["language_instruction"][()]
    context.language_instruction = (
        li.decode("utf-8") if isinstance(li, bytes) else str(li)
    )

    context.left1_video_path = os.path.join(
        h5_dir,
        h5_file["image_observations"]["exterior_image_1_left"][()].decode("utf-8"),
    )
    context.left2_video_path = os.path.join(
        h5_dir,
        h5_file["image_observations"]["exterior_image_2_left"][()].decode("utf-8"),
    )
    context.wrist_video_path = os.path.join(
        h5_dir,
        h5_file["image_observations"]["wrist_image_left"][()].decode("utf-8"),
    )
    context.video_paths_list = [
        ("exterior_image_1_left", context.left1_video_path),
        ("exterior_image_2_left", context.left2_video_path),
        ("wrist_image_left", context.wrist_video_path),
    ]


@register_test("check_lengths")
def _check_lengths(context: DataInstance) -> None:
    """
    Validate that observations and actions have the same trajectory length.
    Save the trajectory length in the data class.

    Args:
        context: DataInstance containing observations and actions

    Raises:
        AssertionError: If observations and actions have different lengths
    """
    observations = context.observations
    actions = context.actions

    # Get all trajectory lengths from observations
    obs_lengths = []
    for obs_key, obs in observations.items():
        if hasattr(obs, "shape") and obs.shape is not None and len(obs.shape) > 0:
            obs_lengths.append((obs_key, obs.shape[0]))

    # Get all trajectory lengths from actions
    action_lengths = []
    for action_key, action in actions.items():
        if hasattr(action, "shape") and action.shape is not None and len(action.shape) > 0:
            action_lengths.append((action_key, action.shape[0]))

    # Check the lengths are consistent across all observations and actions
    all_lengths = obs_lengths + action_lengths
    unique_lengths = set(length for _, length in all_lengths)
    assert len(unique_lengths) == 1, (
        f"Inconsistent trajectory lengths across observations and actions: {all_lengths}"
    )

    context.trajectory_length = (
        unique_lengths.pop()
    )  # Save the trajectory length in context
    assert context.trajectory_length > 0, "Trajectory length is zero"

    return True


@register_test("mp4_files_exist_and_readable")
def _validate_mp4_files_exist_and_readable(context: DataInstance) -> None:
    """
    Validate that all MP4 video files exist and are readable.

    Args:
        context: DataInstance containing video file paths

    Raises:
        AssertionError: If any MP4 file does not exist or is not readable
    """
    video_paths = context.video_paths_list or []
    assert video_paths, (
        "No video paths loaded (required_keys should set video_paths_list)"
    )

    for video_name, video_path in video_paths:
        assert os.path.exists(video_path), (
            f"MP4 file does not exist for {video_name}: {video_path}"
        )
        assert os.path.isfile(video_path), (
            f"MP4 path is not a file for {video_name}: {video_path}"
        )

        try:
            with open(video_path, "rb") as f:
                f.read(1)
        except Exception as e:
            raise AssertionError(
                f"MP4 file is not readable for {video_name} at {video_path}. Error: {e}"
            )

    return True


@register_test("video_trajectory_size_test")
def _validate_video_trajectory_sizes(context: DataInstance) -> None:
    """
    Validate that MP4 files can be opened and report min/max image dimensions and video lengths.

    Args:
        context: DataInstance containing video file paths

    Raises:
        AssertionError: If any MP4 file cannot be opened or has invalid dimensions/length
    """
    video_paths = context.video_paths_list or []
    assert video_paths, (
        "No video paths loaded (required_keys should set video_paths_list)"
    )

    min_duration_required = _video_min_duration_seconds(context)
    durations: List[Tuple[str, float]] = []
    frame_counts: List[Tuple[str, int]] = []

    for video_name, video_path in video_paths:
        cap = cv2.VideoCapture(video_path)

        min_width, min_height, max_width, max_height = float("inf"), float("inf"), 0, 0
        min_duration, max_duration = float("inf"), 0

        try:
            assert cap.isOpened(), (
                f"Could not open video file for {video_name}: {video_path}"
            )

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            assert width > 0 and height > 0, (
                f"Invalid dimensions ({width}x{height}) for {video_name}: {video_path}"
            )
            assert fps > 0, f"Invalid FPS ({fps}) for {video_name}: {video_path}"
            assert frame_count > 0, (
                f"Invalid frame count ({frame_count}) for {video_name}: {video_path}"
            )

            duration = frame_count / fps

            assert width >= MIN_IMAGE_SIZE and height >= MIN_IMAGE_SIZE, (
                f"Image size too small for {video_name}: {video_path}"
            )
            assert width <= MAX_IMAGE_SIZE and height <= MAX_IMAGE_SIZE, (
                f"Image size too large for {video_name}: {video_path}"
            )
            assert duration >= min_duration_required, (
                f"Video duration too short ({duration:.2f}s, need >= {min_duration_required:.2f}s) "
                f"for {video_name}: {video_path}"
            )
            assert duration <= MAX_TRAJECTORY_LENGTH, (
                f"Video duration too long ({duration:.2f}s) for {video_name}: {video_path}"
            )

            durations.append((video_name, duration))
            frame_counts.append((video_name, frame_count))

            min_width = min(min_width, width)
            min_height = min(min_height, height)
            max_width = max(max_width, width)
            max_height = max(max_height, height)
            min_duration = min(min_duration, duration)
            max_duration = max(max_duration, duration)

        finally:
            cap.release()

    if len(frame_counts) > 1:
        counts_only = [c for _, c in frame_counts]
        assert max(counts_only) - min(counts_only) <= 1, (
            f"Inconsistent frame counts across cameras: {frame_counts}"
        )

    if context.trajectory_length is not None and frame_counts:
        frame_tolerance = max(5, int(0.1 * context.trajectory_length))
        for video_name, frame_count in frame_counts:
            assert abs(frame_count - context.trajectory_length) <= frame_tolerance, (
                f"Frame count/trajectory mismatch for {video_name}: "
                f"frames={frame_count}, trajectory={context.trajectory_length}"
            )

    if context.control_freq and context.trajectory_length and durations:
        expected_duration = context.trajectory_length / context.control_freq
        tolerance = 0.5
        for video_name, duration in durations:
            assert abs(duration - expected_duration) <= tolerance, (
                f"Duration/control_freq mismatch for {video_name}: "
                f"duration={duration:.2f}s, expected={expected_duration:.2f}s"
            )

    return True


# TODO: Add this back
# @register_test("failure_fields")
def _validate_failure_fields(context: DataInstance) -> None:
    """
    Validate that failure-related fields are present and properly populated.

    Args:
        context: DataInstance containing the h5 file

    Raises:
        AssertionError: If any failure fields are missing or invalid
    """
    annotations = context.annotations

    # Check for failure category (check both spellings)
    has_failure_category = "failure_category" in annotations
    assert has_failure_category, "Missing failure_category field in episode_annotations"

    # Check for failure description
    assert "failure_description" in annotations, (
        "Missing failure_description field in episode_annotations"
    )

    # Check for would_retry
    assert "would_retry" in annotations, (
        "Missing would_retry field in episode_annotations"
    )

    # Validate that the fields are not empty
    failure_category = (
        annotations["failure_category"][()].decode("utf-8")
        if isinstance(annotations["failure_category"][()], bytes)
        else str(annotations["failure_category"][()])
    )
    failure_description = (
        annotations["failure_description"][()].decode("utf-8")
        if isinstance(annotations["failure_description"][()], bytes)
        else str(annotations["failure_description"][()])
    )
    would_retry = annotations["would_retry"][()]

    assert failure_category, "failure_category field is empty"
    assert failure_description, "failure_description field is empty"
    assert isinstance(would_retry, (bool, int, float)), (
        "would_retry field must be boolean or numeric"
    )

    return True


def _close_h5_context(context: DataInstance) -> None:
    hf = getattr(context, "h5_file", None)
    if hf is not None:
        try:
            hf.close()
        except Exception:
            pass
        context.h5_file = None


def _run_validation(context: DataInstance) -> None:
    try:
        _run_registered_tests(context)
    finally:
        _close_h5_context(context)


def validate_h5_file(h5_path: str) -> bool:
    """Run all registered tests on a single HDF5 file (e.g. one episode in a session folder).

    Args:
        h5_path: Absolute or relative path to the .h5 file.

    Returns:
        True if all validations pass.

    Raises:
        AssertionError: If any validation fails.
    """
    resolved = os.path.abspath(os.path.normpath(h5_path))
    context = DataInstance(
        base_path=os.path.dirname(resolved),
        episode_id=os.path.splitext(os.path.basename(resolved))[0],
        h5_path=resolved,
    )
    _run_validation(context)
    return True


def validate_session_dir(session_dir: str) -> int:
    """Validate every ``*.h5`` file in a session directory (non-recursive).

    Returns:
        0 if all files passed, 1 if any file failed or the directory is invalid.
    """
    session_path = os.path.abspath(os.path.normpath(session_dir))
    if not os.path.isdir(session_path):
        print(f"\n✗ Not a directory: {session_path}\n")
        return 1

    h5_files = sorted(
        glob.glob(os.path.join(session_path, "*.h5"))
        + glob.glob(os.path.join(session_path, "*.hdf5"))
    )
    if not h5_files:
        print(f"\n✗ No .h5 or .hdf5 files found in {session_path}\n")
        return 1

    print(f"\nValidating {len(h5_files)} HDF5 file(s) in session: {session_path}\n")
    failures = 0
    for i, path in enumerate(h5_files, 1):
        name = os.path.basename(path)
        print(f"{'=' * 72}\n[{i}/{len(h5_files)}] {name}\n{'=' * 72}")
        try:
            validate_h5_file(path)
            print(f"\n✓ All validation tests passed for {name}\n")
        except AssertionError as e:
            failures += 1
            print(f"\n✗ Validation failed: {e}\n")
        except Exception as e:
            failures += 1
            print(f"\n✗ Unexpected error: {e}\n")

    passed = len(h5_files) - failures
    print(f"Session summary: {passed}/{len(h5_files)} file(s) passed.\n")
    return 1 if failures else 0


def validate_policy_data(base_path: str, episode_id: str) -> bool:
    """

    Args:
        base_path: Base path to the dataset (e.g., '/path/to/dataset')
        episode_id: Episode ID to validate (e.g., '000001')

    Returns:
        True if all validations pass

    Raises:
        AssertionError: If any validation fails with descriptive error messages
    """
    context = DataInstance(base_path=base_path, episode_id=episode_id)
    _run_validation(context)

    return True


def main():
    """
    Main function to run validation tests from command line.

    Usage:
        python validate.py --samples_dir /path/to/dataset --episode_id 000001
        python validate.py --samples_dir /path/to/dataset   # validates all *.h5 in folder
    """
    parser = argparse.ArgumentParser(
        description="Validate robotic failure dataset episodes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--samples_dir",
        "-o",
        type=str,
        required=True,
        help="Directory containing formatted episode files",
    )

    parser.add_argument(
        "--episode_id",
        "-e",
        type=str,
        required=False,
        default=None,
        help="Episode ID to validate (e.g., 000001); if omitted, all *.h5 files in samples_dir are validated",
    )

    parser.add_argument(
        "--gap_analysis",
        "-g",
        type=str,
        metavar="SOURCE_H5",
        default=None,
        help="Run gap analysis on a source HDF5 (e.g. ACT/ALOHA) and show required transformations",
    )

    args = parser.parse_args()

    if args.gap_analysis:
        gap_analysis(args.gap_analysis)
        return 0

    if args.episode_id is None:
        return validate_session_dir(args.samples_dir)

    try:
        print(f"\nValidating episode {args.episode_id} from {args.samples_dir}...\n")
        validate_policy_data(args.samples_dir, args.episode_id)
        print(f"\n✓ All validation tests passed for episode {args.episode_id}\n")
        return 0

    except AssertionError as e:
        print(f"\n\n✗ Validation failed: {e}\n")
        return 1

    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}\n")
        return 1


if __name__ == "__main__":
    exit(main())
