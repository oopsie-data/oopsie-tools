"""Public validation API.

Composes episode_loader (file I/O, schema detection, video loading) and
episode_validator (semantic checks on loaded data) into the entry points
used by the CLI and tests.
"""

from __future__ import annotations

import glob
import os

from oopsie_tools.utils.validation.episode_loader import load_episode_from_h5
from oopsie_tools.utils.validation.episode_validator import validate_episode


def validate_h5_file(h5_path: str, strict_annotation_check: bool = False) -> bool:
    """Validate a single HDF5 episode file.

    Args:
        h5_path: Path to the .h5 file.
        strict_annotation_check: If True, require that annotations are present and non-empty.

    Returns:
        True if all checks pass.

    Raises:
        AssertionError: On the first validation failure.
    """
    data = load_episode_from_h5(h5_path)
    validate_episode(data, strict_annotation_check=strict_annotation_check)
    return True


def validate_session_dir(session_dir: str, strict_annotation_check: bool = False ) -> int:
    """Validate every ``*.h5`` / ``*.hdf5`` file in a session directory.

    Returns:
        0 if all files passed, 1 if any failed or the directory is invalid.
    """
    session_path = os.path.abspath(os.path.normpath(session_dir))
    if not os.path.isdir(session_path):
        print(f"\n✗ Not a directory: {session_path}\n")
        return 1

    # find all hdf5 files recursively in the session directory
    h5_files = [
        f
        for ext in ("*.h5", "*.hdf5")
        for f in glob.glob(os.path.join(session_path, "**", ext), recursive=True)
    ]

    if not h5_files:
        print(f"\n✗ No .h5 or .hdf5 files found in {session_path}\n")
        return 1

    print(f"\nValidating {len(h5_files)} HDF5 file(s) in: {session_path}\n")
    failures = 0
    for i, path in enumerate(h5_files, 1):
        name = os.path.basename(path)
        print(f"{'=' * 72}\n[{i}/{len(h5_files)}] {name}\n{'=' * 72}")
        try:
            validate_h5_file(path, strict_annotation_check=strict_annotation_check)
            print(f"\n✓ {name} passed\n")
        except AssertionError as e:
            failures += 1
            print(f"\n✗ {name} failed: {e}\n")
        except Exception as e:
            failures += 1
            print(f"\n✗ {name} unexpected error: {e}\n")

    passed = len(h5_files) - failures
    print(f"Summary: {passed}/{len(h5_files)} passed.\n")
    return 1 if failures else 0
