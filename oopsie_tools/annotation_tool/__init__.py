"""Annotation tool for robotic failure samples with multi-user support."""

from pathlib import Path

ANNOTATION_TOOL_DIR = Path(__file__).parent
ANNOTATIONS_FILE = ANNOTATION_TOOL_DIR / "templates/annotations.json"
QUESTIONNAIRE_PATH = ANNOTATION_TOOL_DIR / "templates/questionnaire.yaml"
ROBOT_EMBODIMENTS_PATH = ANNOTATION_TOOL_DIR / "templates/robot_embodiments.txt"
POLICIES_PATH = ANNOTATION_TOOL_DIR / "templates/policies.txt"

# VLM model abbreviations for annotator names
VLM_ABBREVIATIONS = {
    "nvidia/Cosmos-Reason1-7B": "cosmos-7b",
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen-7b",
}


def get_vlm_annotator_name(model: str) -> str:
    """Get abbreviated annotator name for VLM model.

    Args:
        model: Model name (CLI name or full model ID)

    Returns:
        Abbreviated annotator name (e.g., "cosmos-7b")
    """
    # Map CLI model names to full names
    model_full = {
        "cosmos": "nvidia/Cosmos-Reason1-7B",
        "cosmos-reason1": "nvidia/Cosmos-Reason1-7B",
        "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
        "qwen-vl": "Qwen/Qwen2.5-VL-7B-Instruct",
    }.get(model, model)
    return VLM_ABBREVIATIONS.get(model_full, model.split("/")[-1].lower())
