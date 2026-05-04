"""Validate HDF5 episodes against the oopsiedata schema.

Usage:
    python validate.py /path/to/session_dir          # all *.h5 in directory
    python validate.py /path/to/episode.h5           # single episode file
"""

import argparse
import os
import sys

from oopsie_tools.utils.validation.validation_utils import validate_h5_file, validate_session_dir


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate oopsie episode HDF5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a single .h5 file or a session directory containing .h5 files",
    )
    parser.add_argument(
        "--strict_annotation_check",
        action="store_true",
        help="If set, require that annotations are present and non-empty (recommended for upload validation)",
    )
    args = parser.parse_args()
    target = os.path.abspath(os.path.normpath(args.path))

    if os.path.isfile(target):
        try:
            validate_h5_file(target, strict_annotation_check=args.strict_annotation_check)
            print(f"\n✓ {os.path.basename(target)} passed\n")
            return 0
        except AssertionError as e:
            print(f"\n✗ Validation failed: {e}\n")
            return 1
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}\n")
            return 1

    if os.path.isdir(target):
        return validate_session_dir(target)

    print(f"\n✗ Path does not exist: {target}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
