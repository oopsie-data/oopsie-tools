#!/bin/bash
# Launch the video annotation tool with multi-user support
# Usage: ./launch_annotator.sh [--samples-dir DIR] [--port PORT] [--no-browser]
#
# This is a convenience wrapper. The tool is located at:
# oopsie_tools/annotation_tool/launch.sh
#
# Ensure we run from the project root so relative paths are consistent.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Set UV cache directory
export UV_CACHE_DIR="$HOME/.cache/uv"

uv run python -m oopsie_tools.annotation_tool.annotator_server "$@"
