"""Minimal rollout-side client for the minimal annotation server.

The rollout annnotator is responsible for:
- waiting for a language instruction
- recording step data
- after rollout: saving MP4s, show them in the web UI, wait for Save
- write annotation into episode HDF5 via EpisodeRecorder (failure_annotation attr)
"""

from __future__ import annotations

import atexit
import datetime
import json
import os
import socket
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
from moviepy.editor import ImageSequenceClip

from oopsie_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_tools.utils.robot_profile import RobotProfile


class WebRolloutAnnotator:
    def __init__(
        self,
        robot_profile: RobotProfile,
        data_root_dir: Path,
        operator_name: str,
        annotator_name: str | None = None,
        port: int = 5001,
        wait_for_annotation: bool = True,
        open_browser: bool = True,
        resume_session_name: str | None = None,
    ) -> None:
        self.robot_profile = robot_profile
        self.data_root_dir = data_root_dir.resolve()
        self.operator_name = operator_name.strip()
        self.annotator_name = annotator_name.strip() if annotator_name is not None else self.operator_name
        self.port = port
        self.wait_for_annotation = wait_for_annotation
        self.open_browser = open_browser
        self._proc: subprocess.Popen | None = None
        self._active_recorder = EpisodeRecorder(
            robot_profile=self.robot_profile,
            data_root_dir=str(self.data_root_dir),
            resume_session_name=resume_session_name,
            operator_name=self.operator_name,
        )

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self.data_root_dir.mkdir(parents=True, exist_ok=True)

        if not self._is_port_open():
            self._proc = subprocess.Popen(
                [
                    "python",
                    "-m",
                    "oopsie_tools.annotation_tool.annotator_server",
                    "--samples-dir",
                    str(self.data_root_dir),
                    "--port",
                    str(self.port),
                    "--annotator-name",
                    self.annotator_name,
                    "--with-rollouts",
                    "--no-browser",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            atexit.register(self.stop)

            for _ in range(40):
                time.sleep(0.25)
                if self._is_port_open():
                    break

        if self.open_browser:
            webbrowser.open(f"http://localhost:{self.port}/")

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.terminate()
            self._proc = None

    def _is_port_open(self) -> bool:
        with socket.socket() as s:
            return s.connect_ex(("localhost", self.port)) == 0

    # ------------------------------------------------------------------
    # Task coordination
    # ------------------------------------------------------------------

    def wait_for_task(self) -> str:
        while True:
            try:
                state = self._api_get("/api/task/state")
                if (
                    isinstance(state, dict)
                    and state.get("status") == "pending"
                    and state.get("pending_instruction")
                ):
                    instruction = str(state["pending_instruction"])
                    self._api_post("/api/task/start", {"instruction": instruction})
                    return instruction
            except Exception:
                pass
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Optional recording passthrough
    # ------------------------------------------------------------------

    def reset_episode_recorder(self) -> None:
        if self._active_recorder is not None:
            self._active_recorder.reset_episode_recorder()

    def record_step(self, observation: dict[str, Any], action: dict[str, Any]) -> None:
        if self._active_recorder is not None:
            self._active_recorder.record_step(observation=observation, action=action)

    # ------------------------------------------------------------------
    # Post-rollout workflow
    # ------------------------------------------------------------------

    def finish_rollout(
        self,
        instruction: str,
        recorder: EpisodeRecorder | None = None,
    ) -> dict[str, Any] | None:
        assert isinstance(instruction, str) and instruction.strip(), f"Instruction must be a non-empty string. got {instruction}"

        active_recorder = recorder or self._active_recorder
        if active_recorder is None:
            raise ValueError("EpisodeRecorder is required (pass recorder=...).")
        sample_id = self._active_recorder.save_fname

        # 1. Save videos under the recorder's per-session folder
        video_paths = active_recorder._save_videos()
        video_urls = {
            cam: self._video_url_from_abs_path(Path(p))
            for cam, p in video_paths.items()
        }

        # Save the episode HDF5 immediately after rollout (before annotation arrives).
        # We will patch failure_annotation in-place later.
        #
        # IMPORTANT: ensure camera_names is populated so EpisodeRecorder writes
        # `image_observations/<cam>` datasets pointing at MP4 filepaths.
        if not getattr(active_recorder, "camera_names", None):
            active_recorder.camera_names = list(video_paths.keys())

        # 2. Save the episode HDF5 immediately after rollout to disk (before annotation arrives).
        h5_path = active_recorder.save(
            {
                "language_instruction": instruction,
                "metadata": {
                    "episode_id": sample_id,
                    "operator_name": self.operator_name,
                },
                "video_paths": video_paths,
            }
        )

        # 3. Wait for annotation from human annotator
        self._api_post(
            "/api/task/annotating",
            {
                "sample_id": sample_id,
                "video_urls": video_urls,
                "language_instruction": instruction,
            },
        )
        annotation: dict[str, Any] | None = None
        if self.wait_for_annotation:
            annotation = self._wait_for_annotation(sample_id)
        # Reset UI to idle regardless of save/write outcome
        self._api_post("/api/task/done", {})

        # 4. Write annotation into episode HDF5 in-place
        if annotation:
            # Mirror the old behavior of also writing annotation into a sidecar JSON
            # to keep it co-located with the MP4s.
            sample_json = active_recorder.session_dir / f"{sample_id}.json"
            metadata = {
                "language_instruction": instruction,
                "camera_names": list(video_paths.keys()),
                "timestamp": datetime.datetime.now().isoformat(),
                "annotation": {
                    k: v
                    for k, v in annotation.items()
                    if k not in {"annotated_at", "annotator", "source"}
                },
            }
            fd, tmp = tempfile.mkstemp(
                dir=active_recorder.session_dir, suffix=".json.tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                Path(tmp).replace(sample_json)
            except Exception:
                try:
                    Path(tmp).unlink(missing_ok=True)
                except Exception:
                    pass
                raise

            # Patch the already-saved episode HDF5 in-place
            EpisodeRecorder.patch_h5_failure_annotation(Path(h5_path), annotation)

        return annotation

    def _save_videos(
        self,
        output_dir: Path,
        sample_id: str,
        videos: dict[str, list[np.ndarray]],
    ) -> tuple[str, dict[str, str]]:
        video_paths: dict[str, str] = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        for cam_name, frames in videos.items():
            video_path = output_dir / f"{sample_id}_{cam_name}.mp4"
            ImageSequenceClip(list(np.stack(frames)), fps=10).write_videofile(
                str(video_path),
                codec="libx264",
                logger=None,
            )
            video_paths[cam_name] = str(video_path.resolve())
        return sample_id, video_paths

    def _wait_for_annotation(self, sample_id: str) -> dict[str, Any]:
        while True:
            data = self._api_get(f"/api/annotations/{sample_id}")
            if not isinstance(data, dict):
                time.sleep(0.5)
                continue
            if data.get("__annotation_skipped__"):
                return {}
            inner = (
                data.get("annotation")
                if isinstance(data.get("annotation"), dict)
                else None
            )
            if inner and inner.get("__annotation_skipped__"):
                return {}
            if data:
                # minimal server returns the annotation dict directly
                return inner if inner is not None else data
            time.sleep(0.5)

    def _video_url_from_abs_path(self, abs_path: Path) -> str:
        rel = abs_path.resolve().relative_to(self.data_root_dir.resolve()).as_posix()
        return f"/videos-path/{urllib.parse.quote(rel, safe='/')}"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _api_get(self, path: str) -> Any:
        url = f"http://localhost:{self.port}{path}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            return json.loads(resp.read())

    def _api_post(self, path: str, data: dict[str, Any]) -> Any:
        url = f"http://localhost:{self.port}{path}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())
