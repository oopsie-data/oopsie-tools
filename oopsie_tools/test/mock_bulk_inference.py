#!/usr/bin/env python3
"""Mock bulk policy inference run using EpisodeRecorder.

Simulates a policy evaluation loop: for each episode a "policy" generates
random actions, the EpisodeRecorder collects observations and saves to HDF5,
then validate_session_dir is called on all produced files.

Usage:
    uv run python scripts/mock_bulk_inference.py
    uv run python scripts/mock_bulk_inference.py --out-dir /tmp/my_session
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Add validate script to path so it can be imported without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts/validate_and_upload"))
from validate import validate_session_dir  # noqa: E402

from oopsie_tools.annotation_tool.episode_recorder import EpisodeRecorder
from oopsie_tools.utils.robot_profile.robot_profile import RobotProfile


# ── Robot configuration ───────────────────────────────────────────────────────

PROFILE = RobotProfile(
    policy_name="mock_pi0",
    robot_name="franka_research_3",
    is_biarm=False,
    uses_mobile_base=False,
    gripper_name="robotiq_2f_85",
    control_freq=10,
    camera_names=["front", "wrist"],
    robot_state_keys=["joint_position", "gripper_position"],
    robot_state_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
    action_space=["joint_velocity", "gripper_position"],
    action_joint_names=["j1", "j2", "j3", "j4", "j5", "j6", "j7"],
)

EPISODES = [
    {"instruction": "pick up the red block", "n_steps": 25, "success": 1.0},
    {"instruction": "place the cup on the tray", "n_steps": 30, "success": 0.0},
    {"instruction": "open the drawer", "n_steps": 25, "success": 1.0},
    {"instruction": "stack the blocks", "n_steps": 28, "success": 0.0},
    {"instruction": "move arm to home position", "n_steps": 22, "success": 1.0},
]

IMG_H, IMG_W = 224, 224


# ── Mock policy and observation ───────────────────────────────────────────────

class MockPolicy:
    """Generates random joint-velocity + gripper-position actions."""

    def __call__(self, obs: dict) -> dict:  # noqa: ARG002
        return {
            "joint_velocity": (np.random.randn(7) * 0.05).astype(np.float32),
            "gripper_position": np.clip(
                np.random.randn(1).astype(np.float32), -1.0, 1.0
            ),
        }


def _make_obs(step: int) -> dict:
    rng = np.random.default_rng(step)
    return {
        "robot_state": {
            "joint_position": rng.standard_normal(7).astype(np.float32),
            "gripper_position": np.array([rng.uniform(-1, 1)], dtype=np.float32),
        },
        "image_observation": {
            cam: rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)
            for cam in PROFILE.camera_names
        },
    }


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(out_dir: Path) -> int:
    print(f"\nOutput directory: {out_dir}")
    policy = MockPolicy()
    recorder = EpisodeRecorder(
        robot_profile=PROFILE,
        data_root_dir=out_dir,
        operator_name="mock_operator",
    )
    session_dir = recorder.session_dir
    print(f"Session: {session_dir.name}")
    print(f"Recording {len(EPISODES)} episodes...\n")

    for ep_idx, cfg in enumerate(EPISODES, 1):
        recorder.reset_episode_recorder()
        n = cfg["n_steps"]
        instruction = cfg["instruction"]
        success = cfg["success"]

        t0 = time.perf_counter()
        print(f"  [{ep_idx}/{len(EPISODES)}] '{instruction}'  ({n} steps)", end="", flush=True)

        for step in range(n):
            obs = _make_obs(step)
            action = policy(obs)
            recorder.record_step(obs, action)

        recorder.finish_rollout(instruction=instruction, success=success)
        elapsed = time.perf_counter() - t0
        print(f"  →  saved  ({elapsed:.1f}s)")

        # Avoid same-second filename collisions when steps complete very fast.
        if ep_idx < len(EPISODES):
            time.sleep(1.1)

    print(f"\n{'=' * 62}")
    print("Validation")
    print(f"{'=' * 62}")
    result = validate_session_dir(str(session_dir))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory to write session data (default: a temp dir that persists after the run)",
    )
    args = parser.parse_args()

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        return run(args.out_dir)

    # Use a non-auto-deleted temp dir so the user can inspect output.
    tmp = tempfile.mkdtemp(prefix="mock_bulk_inference_")
    return run(Path(tmp))


if __name__ == "__main__":
    sys.exit(main())
