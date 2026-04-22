#!/usr/bin/env python3
"""Minimal web UI for instruction -> rollout -> annotate -> save.

This is intentionally slimmed down compared to `annotator.py`:
- No session wizard
- No sample indexing / bulk annotation / overview
- No VLM features

Flow:
1) Browser submits a language instruction.
2) External rollout loop polls the instruction, runs policy, saves MP4s & HDF5, then
   signals this server to show the videos + questionnaire.
3) Browser fills questionnaire + clicks Save.
4) Rollout loop polls the saved annotation and writes it into the episode HDF5
   (via EpisodeRecorder).
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import h5py
import numpy as np
import yaml
from flask import Flask, abort, jsonify, request, send_file

from oopsie_tools.annotation_tool import QUESTIONNAIRE_PATH

app = Flask(__name__)


def _json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    return payload if isinstance(payload, dict) else {}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_questionnaire() -> dict[str, Any]:
    if QUESTIONNAIRE_PATH.exists():
        parsed = yaml.safe_load(QUESTIONNAIRE_PATH.read_text(encoding="utf-8"))
        if isinstance(parsed, dict):
            return parsed
    return {"title": "Annotation", "questions": []}


@dataclass
class ServerConfig:
    samples_dir: Path
    annotator_name: str
    # HDF5 browser + questionnaire only (no instruction / rollout cards).
    browse_only: bool = False


class Runtime:
    def __init__(self, cfg: ServerConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.task_state: dict[str, Any] = {
            "status": "idle",  # idle | pending | running | annotating
            "pending_instruction": None,
            "last_instruction": "",
            "current_sample": None,  # dict: {sample_id, video_urls, language_instruction}
        }
        self.annotations: dict[str, Any] = {}  # sample_id -> annotation dict

    def set_pending_instruction(self, instruction: str) -> None:
        with self._lock:
            self.task_state["pending_instruction"] = instruction
            self.task_state["status"] = "pending"

    def start_task(self, instruction: str) -> None:
        with self._lock:
            self.task_state["last_instruction"] = instruction
            self.task_state["pending_instruction"] = None
            self.task_state["status"] = "running"

    def set_annotating(
        self, sample_id: str, video_urls: dict[str, str], language_instruction: str
    ) -> None:
        with self._lock:
            self.task_state["current_sample"] = {
                "sample_id": sample_id,
                "video_urls": video_urls,
                "language_instruction": language_instruction,
            }
            self.task_state["status"] = "annotating"

    def mark_done(self) -> None:
        with self._lock:
            if self.task_state.get("status") == "annotating":
                cs = self.task_state.get("current_sample")
                if isinstance(cs, dict):
                    sid = str(cs.get("sample_id", "")).strip()
                    if sid and sid not in self.annotations:
                        # Lets rollout poll observe "skip" (Back) vs still waiting.
                        self.annotations[sid] = {"__annotation_skipped__": True}
            self.task_state["status"] = "idle"
            self.task_state["current_sample"] = None

    def save_annotation(
        self, sample_id: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        reserved = {"annotated_at", "annotator", "source", "schema", "__annotation_skipped__"}
        annotation = {k: v for k, v in payload.items() if k not in reserved}
        annotation.update(
            {
                "annotated_at": _now_iso(),
                "annotator": self.cfg.annotator_name,
                "source": "human",
                "schema": "oopsie_failure_taxonomy_v1",
            }
        )
        with self._lock:
            self.annotations[sample_id] = annotation
        return annotation

    def get_annotation(self, sample_id: str) -> dict[str, Any] | None:
        with self._lock:
            ann = self.annotations.get(sample_id)
            return ann if isinstance(ann, dict) else None


def _get_runtime() -> Runtime:
    rt = app.extensions.get("annotator_runtime")
    if isinstance(rt, Runtime):
        return rt
    raise RuntimeError("Runtime not configured. Call configure_runtime().")


def configure_runtime(
    samples_dir: Path,
    annotator_name: str,
    *,
    browse_only: bool = False,
) -> Runtime:
    cfg = ServerConfig(
        samples_dir=samples_dir.resolve(),
        annotator_name=annotator_name,
        browse_only=browse_only,
    )
    rt = Runtime(cfg)
    app.extensions["annotator_runtime"] = rt
    return rt


@app.get("/")
def index() -> str:
    html = _load_template("annotator.html")
    browse = "true" if _get_runtime().cfg.browse_only else "false"
    name_js = json.dumps(_get_runtime().cfg.annotator_name)
    html = html.replace(
        "<script>",
        f"<script>\n      const ANNOTATOR_BROWSE_ONLY = {browse};\n"
        f"      const ANNOTATOR_NAME = {name_js};\n",
        1,
    )
    return html


@app.get("/api/task/state")
def api_task_state():
    rt = _get_runtime()
    # Let the UI distinguish rollout idle vs annotation-only (browse) idle.
    return jsonify({**rt.task_state, "browse_only": rt.cfg.browse_only})


@app.post("/api/task/submit")
def api_task_submit():
    instruction = str(_json_payload().get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    _get_runtime().set_pending_instruction(instruction)
    return jsonify({"status": "ok"})


@app.post("/api/task/start")
def api_task_start():
    instruction = str(_json_payload().get("instruction", "")).strip()
    if not instruction:
        return jsonify({"error": "instruction is required"}), 400
    _get_runtime().start_task(instruction)
    return jsonify({"status": "ok"})


@app.post("/api/task/annotating")
def api_task_annotating():
    payload = _json_payload()
    sample_id = str(payload.get("sample_id", "")).strip()
    video_urls = payload.get("video_urls", {})
    language_instruction = str(payload.get("language_instruction", "")).strip()
    if not sample_id:
        return jsonify({"error": "sample_id is required"}), 400
    if not isinstance(video_urls, dict) or not video_urls:
        return jsonify({"error": "video_urls must be a non-empty object"}), 400
    _get_runtime().set_annotating(sample_id, video_urls, language_instruction)
    return jsonify({"status": "ok"})


@app.post("/api/task/done")
def api_task_done():
    _get_runtime().mark_done()
    return jsonify({"status": "ok"})


@app.get("/api/questionnaire")
def api_questionnaire():
    return jsonify(load_questionnaire())


@app.get("/api/annotations/<sample_id>")
def api_get_annotation(sample_id: str):
    ann = _get_runtime().get_annotation(sample_id)
    return jsonify(ann or {})


@app.post("/api/annotations")
def api_save_annotation_json() -> Any:
    """Save annotation with ``sample_id`` in the JSON body (avoids slashes in URL paths)."""
    payload = _json_payload()
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid JSON body"}), 400
    sample_id = str(payload.get("sample_id", "")).strip()
    if not sample_id:
        return jsonify({"error": "sample_id is required"}), 400
    answers = {k: v for k, v in payload.items() if k != "sample_id"}
    ann = _get_runtime().save_annotation(sample_id, answers)
    return jsonify({"status": "saved", "annotation": ann})


@app.post("/api/annotations/<sample_id>")
def api_save_annotation(sample_id: str):
    payload = _json_payload()
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid JSON body"}), 400
    ann = _get_runtime().save_annotation(sample_id, payload)
    return jsonify({"status": "saved", "annotation": ann})


@app.get("/videos-path/<path:video_path>")
def serve_video_by_path(video_path: str):
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    target_path = (samples_root / video_path).resolve()
    try:
        target_path.relative_to(samples_root)
    except ValueError:
        abort(403)
    if not target_path.exists() or not target_path.is_file():
        abort(404)
    if target_path.suffix.lower() == ".mp4":
        return send_file(target_path, mimetype="video/mp4")
    return send_file(target_path)


def _video_url_for_path(samples_dir: Path, abs_path: Path) -> str:
    rel = abs_path.resolve().relative_to(samples_dir.resolve()).as_posix()
    return f"/videos-path/{quote(rel, safe='/')}"


def _relative_to_samples_dir(samples_root: Path, path: Path) -> str | None:
    try:
        return path.resolve().relative_to(samples_root.resolve()).as_posix()
    except ValueError:
        return None


def _resolve_video_path(
    samples_root: Path, raw_path: str, h5_path: Path
) -> Path | None:
    candidate = Path(raw_path)

    if candidate.is_absolute():
        if candidate.exists():
            return candidate
    else:
        local_candidate = (h5_path.parent / candidate).resolve()
        if local_candidate.exists():
            return local_candidate

    basename_candidate = (h5_path.parent / candidate.name).resolve()
    if basename_candidate.exists():
        return basename_candidate

    samples_root_candidate = (samples_root.resolve() / candidate.name).resolve()
    if samples_root_candidate.exists():
        return samples_root_candidate

    return None


def _decode_h5_value(value: Any) -> Any:
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")

    if hasattr(value, "shape") and getattr(value, "shape", None) == ():
        try:
            return _decode_h5_value(value.item())
        except Exception:
            return value

    if hasattr(value, "tolist"):
        try:
            listed = value.tolist()
            if isinstance(listed, list):
                return [_decode_h5_value(v) for v in listed]
            return _decode_h5_value(listed)
        except Exception:
            pass

    return value


def _read_h5_attr(group: h5py.Group | h5py.File, key: str, default: Any = "") -> Any:
    return _decode_h5_value(group.attrs.get(key, default))


class H5PathError(ValueError):
    """Invalid or unsafe HDF5 path from query string; carries HTTP status for API responses."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


def _safe_h5_path_from_query(samples_root: Path) -> Path:
    raw = str(request.args.get("path", "")).strip()
    if not raw:
        raise H5PathError("path query parameter is required", 400)
    rel = Path(unquote(raw))
    target = (samples_root.resolve() / rel).resolve()
    try:
        target.relative_to(samples_root.resolve())
    except ValueError:
        raise H5PathError("path escapes samples directory", 403)
    if target.suffix.lower() != ".h5":
        raise H5PathError("path must refer to an .h5 file", 400)
    if not target.exists() or not target.is_file():
        raise H5PathError("HDF5 file not found", 404)
    return target


def _binary_success_from_success_attr(val: Any) -> str | None:
    """Map ``episode_annotations/<annotator>.attrs['success']`` float to questionnaire values."""
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return "Success" if f >= 0.5 else "Failure"


def _parse_taxonomy_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    s = str(raw or "").strip()
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except json.JSONDecodeError:
        return {}


def _annotator_subgroup_looks_annotated(fa: h5py.Group) -> bool:
    src = str(_read_h5_attr(fa, "source", "") or "").strip().lower()
    if src == "human":
        return True
    if _read_h5_attr(fa, "success", None) is not None:
        return True
    if str(_read_h5_attr(fa, "failure_description", "") or "").strip():
        return True
    if str(_read_h5_attr(fa, "taxonomy", "") or "").strip():
        return True
    return False


@app.get("/api/h5/list")
def api_h5_list():
    rt = _get_runtime()
    root = rt.cfg.samples_dir.resolve()
    entries: list[dict[str, Any]] = []
    for p in root.rglob("*.h5"):
        if not p.is_file():
            continue
        rel = p.resolve().relative_to(root).as_posix()
        annotated = False
        try:
            with h5py.File(p, "r") as h5f:
                ea = h5f.get("episode_annotations")
                if isinstance(ea, h5py.Group):
                    raw = _read_h5_attr(ea, "failure_annotation", "")
                    raw_s = str(raw or "").strip()
                    if raw_s:
                        try:
                            parsed = json.loads(raw_s)
                            # Treat empty JSON object `{}` as not annotated.
                            if isinstance(parsed, dict):
                                annotated = len(parsed) > 0
                            else:
                                # Any other valid JSON (e.g. string/list) counts as annotated if non-empty.
                                annotated = bool(parsed)
                        except Exception:
                            # Non-JSON payload (legacy / free-text) counts as annotated if non-empty.
                            annotated = True
                    if not annotated:
                        ann = rt.cfg.annotator_name
                        if ann in ea.keys() and isinstance(ea[ann], h5py.Group):
                            annotated = _annotator_subgroup_looks_annotated(ea[ann])
        except Exception:
            annotated = False
        entries.append(
            {
                "id": rel.replace("/", "__"),
                "rel_path": rel,
                "display_name": p.name,
                "mtime": p.stat().st_mtime,
                "annotated": annotated,
            }
        )
    entries.sort(key=lambda e: e.get("rel_path") or "")
    return jsonify(entries)


@app.get("/api/h5/sample")
def api_h5_sample():
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    try:
        h5_path = _safe_h5_path_from_query(samples_root)
    except H5PathError as e:
        return jsonify({"error": e.message}), e.status

    video_urls: dict[str, str] = {}
    existing_annotation: dict[str, Any] = {}
    metadata: dict[str, Any] = {
        "rel_path": h5_path.resolve().relative_to(samples_root).as_posix()
    }

    with h5py.File(h5_path, "r") as h5f:
        metadata["language_instruction"] = (
            str(_read_h5_attr(h5f, "language_instruction", "")) or ""
        )
        metadata["episode_id"] = str(_read_h5_attr(h5f, "episode_id", "")) or ""
        metadata["operator_name"] = str(_read_h5_attr(h5f, "operator_name", "")) or ""

        ea = h5f.get("episode_annotations")
        if isinstance(ea, h5py.Group):
            ann_name = rt.cfg.annotator_name
            if ann_name in ea.keys() and isinstance(ea[ann_name], h5py.Group):
                fa = ea[ann_name]
                metadata["success"] = _read_h5_attr(fa, "success", None)
                succ = _read_h5_attr(fa, "success", None)
                bs_label = _binary_success_from_success_attr(succ)
                if bs_label is not None:
                    existing_annotation["binary_success"] = bs_label
                existing_annotation["failure_description"] = str(
                    _read_h5_attr(fa, "failure_description", "") or ""
                )
                existing_annotation["additional_notes"] = str(
                    _read_h5_attr(fa, "additional_notes", "") or ""
                )
                tax = _parse_taxonomy_json(_read_h5_attr(fa, "taxonomy", ""))
                fc = tax.get("failure_category")
                if fc is not None:
                    if isinstance(fc, (list, tuple)):
                        existing_annotation["failure_category"] = [str(x) for x in fc]
                    else:
                        existing_annotation["failure_category"] = [str(fc)]
                sev = tax.get("severity")
                if sev is not None:
                    existing_annotation["severity"] = str(sev)

            if not existing_annotation:
                raw_failure = _read_h5_attr(ea, "failure_annotation", "")
                raw_s = str(raw_failure or "").strip()
                if raw_s:
                    try:
                        parsed = json.loads(raw_s)
                        if isinstance(parsed, dict) and parsed:
                            skip = {"annotated_at", "annotator", "source"}
                            existing_annotation = {
                                k: v for k, v in parsed.items() if k not in skip
                            }
                    except Exception:
                        pass

        # breakpoint()
        image_group = h5f.get("image_observations")
        if isinstance(image_group, h5py.Group):
            for cam in image_group.keys():
                try:
                    raw_path = _decode_h5_value(image_group[cam][()])
                except Exception:
                    continue
                if not isinstance(raw_path, str) or not raw_path:
                    continue
                vp = _resolve_video_path(samples_root, raw_path, h5_path)
                if vp is None:
                    continue
                rel = _relative_to_samples_dir(samples_root, vp)
                if rel is None:
                    continue
                video_urls[str(cam)] = f"/videos-path/{quote(rel, safe='/')}"

    # Fallback: if HDF5 doesn't store `image_observations/*` paths, try to infer
    # videos from files next to the H5: `<stem>_<cam>.mp4`.
    if not video_urls:
        stem = h5_path.stem
        for mp4 in sorted(h5_path.parent.glob(f"{stem}_*.mp4")):
            cam = mp4.stem[len(stem) + 1 :]  # after "<stem>_"
            if not cam:
                continue
            rel = _relative_to_samples_dir(samples_root, mp4)
            if rel is None:
                continue
            video_urls[cam] = f"/videos-path/{quote(rel, safe='/')}"

    return jsonify(
        {
            "rel_path": metadata["rel_path"],
            "metadata": metadata,
            "video_urls": video_urls,
            "existing_annotation": existing_annotation,
        }
    )


@app.post("/api/h5/annotations")
def api_h5_save_annotation():
    rt = _get_runtime()
    samples_root = rt.cfg.samples_dir.resolve()
    try:
        h5_path = _safe_h5_path_from_query(samples_root)
    except H5PathError as e:
        return jsonify({"error": e.message}), e.status

    payload = _json_payload()
    if not isinstance(payload, dict):
        return jsonify({"error": "invalid JSON body"}), 400

    ann = rt.save_annotation(sample_id=str(h5_path), payload=payload)

    success: float | None = None
    bs = str(ann.get("binary_success", "")).strip().lower()
    if bs == "success":
        success = 1.0
    elif bs == "failure":
        success = 0.0

    try:
        with h5py.File(h5_path, "r+") as f:
            ea = f.require_group("episode_annotations")
            ea.attrs["failure_annotation"] = json.dumps(dict(ann), ensure_ascii=False)
            if success is not None:
                ea.attrs["success"] = float(np.float32(success))
    except OSError as e:
        return jsonify({"error": f"could not write HDF5: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"could not update annotation: {e}"}), 500

    return jsonify({"status": "saved", "annotation": ann})


TEMPLATE_DIR = Path(__file__).parent / "ui"


def _load_template(filename: str) -> str:
    path = (TEMPLATE_DIR / filename).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing template: {path}")
    return path.read_text(encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Annotator server")
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=Path("samples"),
        help="Directory containing saved MP4s (and HDF5 episodes) for this session",
    )
    parser.add_argument("--port", type=int, default=5001)
    parser.add_argument(
        "--annotator-name",
        type=str,
        required=True,
        help="Annotator name to stamp into saved annotations",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument(
        "--with-rollouts",
        action="store_true",
        help="HDF5 browser + questionnaire + rollouts",
    )
    args = parser.parse_args()

    samples_dir = args.samples_dir.resolve()
    samples_dir.mkdir(parents=True, exist_ok=True)
    configure_runtime(
        samples_dir=samples_dir,
        annotator_name=args.annotator_name.strip(),
        browse_only=bool(not args.with_rollouts),
    )

    if not args.no_browser and os.environ.get("WERKZEUG_RUN_MAIN") != "true":
        webbrowser.open(f"http://localhost:{args.port}/")

    # To suppress the werkzeug server logs
    import logging

    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # or logging.ERROR

    app.run(host="localhost", port=args.port, debug=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
