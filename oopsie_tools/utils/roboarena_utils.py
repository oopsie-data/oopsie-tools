from huggingface_hub import snapshot_download

import cv2
import h5py
import json
import numpy as np
import os
import shutil
import logging
import glob

from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def download_raw():
    repo_id = "RoboArena/DataDump_08-05-2025"

    # Download the entire dataset
    local_path = snapshot_download(repo_id=repo_id, repo_type="dataset")

    print(f"Dataset files downloaded to: {local_path}")


def _extract_frames_to_numpy(
    video_path: str, target_size: tuple[int, int] = (180, 320)
) -> np.ndarray:
    """Extract all frames from a video file and return as a numpy array.

    Args:
        video_path: Path to the mp4 video file
        target_size: (height, width) to resize frames to. Default (180, 320).

    Returns:
        np.ndarray of shape (num_frames, height, width, channels)
    """
    vidcap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        success, image = vidcap.read()
        if not success:
            break
        # Resize frame (cv2.resize takes (width, height))
        image_resized = cv2.resize(
            image, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA
        )
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        frames.append(image_rgb)

    vidcap.release()

    if len(frames) == 0:
        log.warning(f"No frames extracted from {video_path}")
        return np.array([])

    return np.stack(frames, axis=0)


def _find_roboarena_data() -> str:
    """Find the RoboArena data directory from HF cache.

    Returns:
        Path to the latest snapshot directory
    """
    hf_base_dir = os.environ["HF_HOME"]
    roboarena_path = "hub/datasets--RoboArena--DataDump_08-05-2025/snapshots"
    data_path = os.path.join(hf_base_dir, roboarena_path)

    snapshots = sorted(glob.glob(os.path.join(data_path, "*")))

    if len(snapshots) == 0:
        raise FileNotFoundError(f"No snapshots found at {data_path}")

    if len(snapshots) > 1:
        log.warning(f"Multiple snapshots found, using the latest: {snapshots[-1]}")

    return snapshots[-1]


def extract_all_frames():
    """Extract frames from all videos in the RoboArena dataset into HDF5 files.

    Creates a 'cleaned' directory above the snapshots folder. Each policy
    directory becomes a single HDF5 file containing:
    - Video frames (gzip compressed, chunked)
    - All data from .npz files

    Non-policy files (e.g., metadata.yaml) are copied as-is.
    """
    snapshot_path = _find_roboarena_data()

    # Create cleaned directory above snapshots
    # Structure: .../datasets--RoboArena--DataDump_08-05-2025/cleaned/
    base_path = os.path.dirname(
        os.path.dirname(snapshot_path)
    )  # Go up two levels from snapshot
    cleaned_path = "/datastor1/droid/roboarena/cleaned"

    log.info(f"Source: {snapshot_path}")
    log.info(f"Destination: {cleaned_path}")

    # Walk through evaluation_sessions
    eval_sessions_path = os.path.join(snapshot_path, "evaluation_sessions")

    if not os.path.exists(eval_sessions_path):
        raise FileNotFoundError(
            f"evaluation_sessions not found at {eval_sessions_path}"
        )

    # Get list of session directories
    session_dirs = [
        d
        for d in os.listdir(eval_sessions_path)
        if os.path.isdir(os.path.join(eval_sessions_path, d))
    ]

    # Iterate through evaluation session directories
    for session_dir in tqdm(session_dirs, desc="Sessions", unit="session"):
        session_path = os.path.join(eval_sessions_path, session_dir)

        # Create corresponding cleaned session directory
        cleaned_session_path = os.path.join(
            cleaned_path, "evaluation_sessions", session_dir
        )
        os.makedirs(cleaned_session_path, exist_ok=True)

        # Process all items in session directory
        for item in tqdm(
            os.listdir(session_path), desc="  Items", unit="item", leave=False
        ):
            item_path = os.path.join(session_path, item)

            if os.path.isfile(item_path):
                # Copy non-mp4 files (e.g., metadata.yaml)
                if not item.endswith(".mp4"):
                    cleaned_item_path = os.path.join(cleaned_session_path, item)
                    shutil.copy2(item_path, cleaned_item_path)

            elif os.path.isdir(item_path):
                # This is a policy directory - create HDF5 file
                h5_path = os.path.join(cleaned_session_path, f"{item}.h5")

                if os.path.exists(h5_path):
                    log.debug(f"Skipping existing: {h5_path}")
                    continue

                _create_policy_hdf5(item_path, h5_path)

    log.info(f"Frame extraction complete. Output at: {cleaned_path}")
    return cleaned_path


def _create_policy_hdf5(policy_dir: str, output_path: str):
    """Create an HDF5 file from a policy directory.

    Args:
        policy_dir: Path to policy directory containing .mp4 and .npz files
        output_path: Path for the output .h5 file
    """
    with h5py.File(output_path, "w") as h5f:
        # Create groups for organization
        frames_group = h5f.create_group("frames")

        for file_name in os.listdir(policy_dir):
            file_path = os.path.join(policy_dir, file_name)

            if file_name.endswith(".mp4"):
                # Extract frames and store with compression
                frames = _extract_frames_to_numpy(file_path)
                if frames.size == 0:
                    continue

                dataset_name = file_name.replace(".mp4", "")
                # Chunk by individual frames for efficient single-frame access
                chunks = (1,) + frames.shape[1:]
                frames_group.create_dataset(
                    dataset_name,
                    data=frames,
                    chunks=chunks,
                    compression="gzip",
                    compression_opts=4,  # Balance between speed and size
                )

            elif file_name.endswith(".npz"):
                # Load npz and store each array in the HDF5
                npz_data = np.load(file_path, allow_pickle=True)
                group_name = file_name.replace(".npz", "")
                npz_group = h5f.create_group(group_name)

                for key in npz_data.files:
                    arr = npz_data[key]

                    # Handle object arrays by serializing to JSON
                    if arr.dtype == object:
                        # Convert to Python object and serialize as JSON string
                        obj = arr.item() if arr.ndim == 0 else arr.tolist()
                        json_str = json.dumps(obj)
                        npz_group.create_dataset(key, data=json_str)
                    # Use compression for larger numeric arrays
                    elif arr.size > 1000:
                        npz_group.create_dataset(
                            key, data=arr, compression="gzip", compression_opts=4
                        )
                    else:
                        npz_group.create_dataset(key, data=arr)

                npz_data.close()


if __name__ == "__main__":
    extract_all_frames()
