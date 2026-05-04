"""Schema-agnostic episode representation used by loader and validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from oopsie_tools.utils.robot_profile.robot_profile import RobotProfile


@dataclass
class VideoInfo:
    """Camera video metadata, constructable from an MP4 file or in-memory frames."""

    frame_count: int
    fps: float
    width: int
    height: int

    @classmethod
    def from_frames(cls, frames: list[np.ndarray], fps: float) -> VideoInfo:
        """Build VideoInfo from in-memory frame buffers (recorder pre-save path)."""
        if not frames:
            raise ValueError("Cannot build VideoInfo from empty frame list")
        h, w = frames[0].shape[:2]
        return cls(frame_count=len(frames), fps=fps, width=w, height=h)


@dataclass
class EpisodeData:
    """Normalized episode data, independent of source schema.

    Populated by episode_loader (from HDF5) or constructed directly by the
    recorder for pre-save validation.  robot_profile is None for legacy schemas
    that do not embed a profile JSON.
    """

    robot_profile: Optional[RobotProfile]
    language_instruction: str
    episode_id: str
    lab_id: str
    operator_name: str
    trajectory_length: int
    control_freq: float
    # Eagerly-loaded arrays; shape (T, ...) for each key.
    observations: dict[str, np.ndarray]
    # Only non-empty action datasets; shape (T, ...) for each key.
    actions: dict[str, np.ndarray]
    # Per-camera metadata.
    videos: dict[str, VideoInfo]
    # Optional: annotator_name → {attr_key: attr_val}.
    annotations: Optional[dict[str, dict[str, Any]]] = field(default=None)
