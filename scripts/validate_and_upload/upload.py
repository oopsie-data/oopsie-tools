"""
End-to-end script: validate formatted robotic failure data and upload to HuggingFace.

Steps:
    1. Authenticate with HuggingFace
    2. Validate the episode(s)
    3. Create HF dataset repo (if it doesn't exist)
    4. Upload dataset files

Usage:
    python upload.py --samples_dir /path/to/formatted_data          # validate all *.h5, upload whole folder
    python upload.py --samples_dir /path/to/formatted_data --episode_id 000001  # single episode

Environment:
    HF_TOKEN  — override the hardcoded token
"""

import sys
import yaml
import os
import argparse
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

# Read lab_id from configs/contributor_config.yaml
try:
    config_path = (
        Path(__file__).resolve().parent.parent.parent
        / "configs"
        / "contributor_config.yaml"
    )
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        lab_id = config.get("lab_id", "").strip()
        huggingface_token = config.get("huggingface_token", "").strip()
        if not lab_id:
            raise ValueError(
                "lab_id must be set in configs/contributor_config.yaml"
            )
except Exception as e:
    raise RuntimeError(
        f"Could not read lab_id from configs/contributor_config.yaml: {e}"
    )

LAB_ID = lab_id
HF_TOKEN = huggingface_token
# TODO: Change this to OopsieData-Submissions/{LAB_ID}
HF_REPO = f"OopsieData-Submissions/{LAB_ID}"

# ── Step 1: HuggingFace authentication ────────────────────────────────────────


def hf_login(token: str):
    from huggingface_hub import login, whoami

    login(token=token, add_to_git_credential=False)
    info = whoami(token=token)
    print(f"[auth]  Logged in as: {info['name']}\n")
    return info["name"]


# ── Step 2: validation ────────────────────────────────────────────────────────


def _validate_import_path():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)


def run_validation(base_path: str, episode_id: str) -> bool:
    _validate_import_path()
    from validate import validate_policy_data  # noqa: E402

    print("[validate] Running validation...")
    try:
        validate_policy_data(base_path, episode_id)
        print("[validate] All checks passed.\n")
        return True
    except AssertionError as e:
        print(f"[validate] FAILED: {e}\n")
        return False
    except Exception as e:
        print(f"[validate] ERROR: {e}\n")
        return False


def run_dir_validation(samples_dir: str) -> bool:
    """Validate every ``*.h5`` in samples_dir."""
    _validate_import_path()
    from validate import validate_session_dir  # noqa: E402

    print("[validate] Running validation (all *.h5 in folder)...")
    code = validate_session_dir(samples_dir)
    if code == 0:
        print("[validate] All files passed.\n")
        return True
    print("[validate] One or more files failed.\n")
    return False


# ── Step 3: create repo (if needed) ───────────────────────────────────────────


def ensure_repo():
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)
    try:
        api.repo_info(repo_id=HF_REPO, repo_type="dataset")
        print(
            f"[hf]    Repo already exists: https://huggingface.co/datasets/{HF_REPO}\n"
        )
    except Exception:
        print(f"[hf]    Creating repo: {HF_REPO}")
        api.create_repo(repo_id=HF_REPO, repo_type="dataset", private=False)
        print(f"[hf]    Created: https://huggingface.co/datasets/{HF_REPO}\n")


# ── Step 4: upload ────────────────────────────────────────────────────────────


def upload_dataset(samples_dir: str, commit_message: str):
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    print(f"[upload] Uploading {samples_dir} → {HF_REPO}")
    print("[upload] Files to upload:")
    total_bytes = 0
    for root, _, files in os.walk(samples_dir):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, samples_dir)
            total_bytes += size
            print(f"           {rel}  ({size / 1e6:.1f} MB)")

    print(f"[upload] Total size: {total_bytes / 1e9:.2f} GB")
    print("[upload] Uploading (this may take several minutes)...")

    api.upload_folder(
        folder_path=samples_dir,
        repo_id=HF_REPO,
        repo_type="dataset",
        commit_message=commit_message,
    )

    print("\n[upload] Done!")
    print(f"[upload] Dataset URL: https://huggingface.co/datasets/{HF_REPO}\n")


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Validate and upload robotic failure dataset to HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--samples_dir",
        "-o",
        required=True,
        help="Base directory containing formatted episode files",
    )
    parser.add_argument(
        "--episode_id",
        "-e",
        default=None,
        help="Episode ID (zero-padded, e.g. 000001); if omitted, all *.h5 files in samples_dir are validated and uploaded",
    )
    parser.add_argument(
        "--skip_validate",
        action="store_true",
        help="Skip validation step before uploading",
    )
    parser.add_argument(
        "--skip_upload", action="store_true", help="Run validation only, do not upload"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Robotic Failure Dataset — End-to-End Upload Pipeline")
    print("=" * 60 + "\n")

    # 1. Auth
    hf_login(HF_TOKEN)

    samples_dir = os.path.abspath(os.path.normpath(args.samples_dir))
    if args.episode_id is None:
        dir_name = os.path.basename(samples_dir.rstrip(os.sep)) or samples_dir
        commit_message = f"Add {dir_name}"
    else:
        commit_message = f"Add episode {args.episode_id}"

    # 2. Validate
    if not args.skip_validate:
        if args.episode_id is None:
            ok = run_dir_validation(samples_dir)
        else:
            ok = run_validation(samples_dir, args.episode_id)
        if not ok:
            print("Aborting upload due to validation failure.")
            print("Fix the dataset format and retry.\n")
            sys.exit(1)
    else:
        print("[validate] Skipped.\n")

    # 3 + 4. Create repo and upload
    if not args.skip_upload:
        ensure_repo()
        upload_dataset(samples_dir, commit_message)
    else:
        print("[upload] Skipped (--skip_upload).\n")


if __name__ == "__main__":
    main()
