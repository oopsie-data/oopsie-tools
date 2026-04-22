"""One-off migration: convert samples from robotic_failure_upload_data_format_v1
to oopsiedata_format_v1 (the layout written by EpisodeRecorder).

Writes a new .h5 alongside each original as <name>.new.h5, then atomically
replaces the original. The original is preserved as <name>.bak.h5.

Usage:
    uv run python scripts/migrate_hdf5_format.py [--samples-dir samples] [--dry-run]
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np


OLD_SCHEMA = "robotic_failure_upload_data_format_v1"
NEW_SCHEMA = "oopsiedata_format_v1"


def _decode(val) -> str:
    if isinstance(val, (bytes, np.bytes_)):
        return val.decode("utf-8")
    return str(val) if val is not None else ""


def _copy_group(src: h5py.Group, dst: h5py.Group) -> None:
    for key in src.keys():
        src.copy(key, dst)
    for k, v in src.attrs.items():
        dst.attrs[k] = v


def migrate_file(src_path: Path, dry_run: bool) -> None:
    bak_path = src_path.with_suffix(".bak.h5")
    tmp_path = src_path.with_suffix(".new.h5")

    try:
        _migrate_file_inner(src_path, tmp_path, bak_path, dry_run)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _migrate_file_inner(src_path: Path, tmp_path: Path, bak_path: Path, dry_run: bool) -> None:
    with h5py.File(src_path, "r") as src:
        schema = _decode(src.attrs.get("schema", ""))
        if schema != OLD_SCHEMA:
            print(f"  skip (schema={schema!r})")
            return

        with h5py.File(tmp_path, "w") as dst:
            str_dtype = h5py.string_dtype(encoding="utf-8")

            # --- root attrs ---
            dst.attrs["schema"] = NEW_SCHEMA

            lang = _decode(src["language_instruction"][()])
            dst.attrs["language_instruction"] = lang

            ea = src["episode_annotations"]
            dst.attrs["episode_id"] = _decode(ea["episode_id"][()]) if "episode_id" in ea else ""
            dst.attrs["operator_name"] = _decode(ea["operator_name"][()]) if "operator_name" in ea else ""
            dst.attrs["lab_id"] = _decode(ea["lab_id"][()]) if "lab_id" in ea else ""

            # --- observations group ---
            obs_grp = dst.create_group("observations")

            # video paths
            vp_grp = obs_grp.create_group("video_paths")
            for cam, ds in src["image_observations"].items():
                vp_grp.create_dataset(cam, data=_decode(ds[()]), dtype=str_dtype)

            # robot states
            rs_grp = obs_grp.create_group("robot_states")
            for key, ds in src["observation"].items():
                rs_grp.create_dataset(key, data=ds[()], dtype=np.float64)

            # --- actions group ---
            act_grp = dst.create_group("actions")
            for key, ds in src["action_dict"].items():
                act_grp.create_dataset(key, data=ds[()], dtype=np.float64)

            # --- episode_annotations: preserve group + annotations ---
            ann_grp = dst.create_group("episode_annotations")
            # copy attrs (failure_annotation JSON, success, etc.)
            for k, v in ea.attrs.items():
                ann_grp.attrs[k] = v
            # copy sub-datasets/groups that aren't already promoted to root attrs
            promoted = {"episode_id", "operator_name", "lab_id"}
            for key in ea.keys():
                if key not in promoted:
                    ea.copy(key, ann_grp)

    if dry_run:
        tmp_path.unlink(missing_ok=True)
        print(f"  dry-run OK")
        return

    shutil.copy2(src_path, bak_path)
    tmp_path.replace(src_path)
    print(f"  migrated (backup: {bak_path.name})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-dir", default="samples", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    h5_files = sorted(
        p for p in args.samples_dir.rglob("*.h5")
        if not p.name.endswith(".bak.h5") and not p.name.endswith(".new.h5")
    )
    if not h5_files:
        print("No .h5 files found.")
        sys.exit(0)

    for path in h5_files:
        print(path)
        migrate_file(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
