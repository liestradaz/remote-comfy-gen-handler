"""Query ComfyUI for available samplers, schedulers, and node info.

Hits the local ComfyUI /object_info endpoint and extracts useful metadata.
"""

import json
import os
import urllib.request

COMFY_HOST = os.environ.get("COMFY_HOST", "127.0.0.1:8188")
COMFY_URL = f"http://{COMFY_HOST}"


def _get_object_info() -> dict:
    """Fetch /object_info from the local ComfyUI instance."""
    with urllib.request.urlopen(f"{COMFY_URL}/object_info", timeout=15) as r:
        return json.loads(r.read())


def _extract_enum_options(node_info: dict, field_name: str) -> list[str]:
    """Extract enum options from a node's input spec."""
    required = node_info.get("input", {}).get("required", {})
    field = required.get(field_name)
    if not field or not isinstance(field, list) or not field:
        return []
    # Enum fields are [[option1, option2, ...]]
    if isinstance(field[0], list):
        return field[0]
    return []


def handle(job: dict) -> dict:
    """Handle a query_info command.

    Expected input:
    {
        "command": "query_info"
    }

    Returns:
    {
        "ok": true,
        "samplers": ["euler", "euler_ancestral", ...],
        "schedulers": ["normal", "karras", ...],
    }
    """
    try:
        object_info = _get_object_info()
    except Exception as e:
        raise RuntimeError(f"Failed to query ComfyUI /object_info: {e}")

    # Extract samplers and schedulers from KSampler node
    ksampler = object_info.get("KSampler", {})
    samplers = _extract_enum_options(ksampler, "sampler_name")
    schedulers = _extract_enum_options(ksampler, "scheduler")

    return {
        "ok": True,
        "samplers": samplers,
        "schedulers": schedulers,
    }
