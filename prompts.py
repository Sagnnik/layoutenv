import json
from typing import Any, Callable, Dict, List, Tuple, Optional

from layoutenv.models import MAGNITUDES, LayoutAction

# Shared body: strategy, metrics, actions — identical for LLM and VLM after each intro.
PROMPT_SHARED_TAIL = """\
Your job is to improve a poster layout step-by-step.  Each step you choose
EXACTLY ONE action that modifies one element.

## Metrics (shown each turn)
- overlap   (lower is better, 0 = none)  — how much elements overlap each other.
- boundary  (lower is better, 0 = none)  — how much elements spill outside the canvas.
- alignment (higher is better, 1 = perfect) — how many element edges/centres line up.
- spacing   (higher is better, 1 = perfect) — how uniform the gaps between elements are.
- plausibility (higher is better, 1 = perfect) — how realistic each element's position
  and size are compared to professional poster layouts.

## Strategy
1. Fix the WORST metric first (highest penalty or lowest score).
2. Prefer LARGE magnitude moves early, SMALL ones to fine-tune.
3. Use ALIGN to snap elements to shared edges for better alignment.
4. Use NO_OP when you are satisfied — it ends the episode.

## Available actions
MOVE   — params: UP, DOWN, LEFT, RIGHT
RESIZE — params: WIDER, NARROWER, TALLER, SHORTER
ALIGN  — params: LEFT, CENTER_X, RIGHT, TOP, CENTER_Y, BOTTOM
SNAP   — params: GRID
NO_OP  — params: NONE

## Magnitudes (for MOVE and RESIZE only)
SMALL  = 0.01     MEDIUM = 0.05 (default)     LARGE = 0.1

## Output format
Return ONLY a valid JSON object on a single line — no explanation, no markdown
fences, no extra text before or after:
{"action": "<ACTION>", "element_id": <int>, "param": "<PARAM>", "magnitude": "<MAG>"}
"""

SYSTEM_PROMPT = """\
You are a UI layout optimisation agent.

""" + PROMPT_SHARED_TAIL

VLM_SYSTEM_PROMPT = """\
You are a UI layout optimisation agent with visual understanding.

You can SEE the poster background image with the current layout rendered on it.
Use this visual context to:
- Avoid placing elements over salient objects or faces
- Position text on clean, low-contrast regions for readability
- Ensure the layout feels harmonious with the visual content

""" + PROMPT_SHARED_TAIL

USER_PROMPT_TEMPLATE = """\
Step {step}/{max_steps}

Current layout:
{elements_json}

Metrics:
  overlap:       {overlap}  (target: 0)
  boundary:      {boundary}  (target: 0)
  alignment:     {alignment}  (target: 1)
  spacing:       {spacing}  (target: 1)
  plausibility:  {plausibility}  (target: 1)
  quality_score: {quality_score}

Feedback: {text_feedback}

Select the best action to improve the layout.
Reply with ONLY the JSON object."""

VLM_USER_PROMPT_TEMPLATE = """\
Step {step}/{max_steps}

You are given an image: the poster background with the CURRENT layout drawn as
coloured boxes and labels. Use it together with the data below to judge
occlusion (whether boxes cover important regions of the artwork) and text
readability against the background.

Current layout (normalised coordinates):
{elements_json}

Metrics:
  overlap:       {overlap}  (target: 0)
  boundary:      {boundary}  (target: 0)
  alignment:     {alignment}  (target: 1)
  spacing:       {spacing}  (target: 1)
  plausibility:  {plausibility}  (target: 1)
  quality_score: {quality_score}

Feedback: {text_feedback}

Visual goals (use the image, not only the scalar metrics):
- Reduce cases where labels or boxes sit on faces, focal objects, or brand marks.
- Move text toward visually calm, reasonably uniform regions when possible.
- Keep the arrangement feeling balanced with the underlying poster composition.

Select the best action to improve the layout.
Reply with ONLY the JSON object."""


def _metrics_format_args(obs_dict: Dict) -> Dict[str, Any]:
    m = obs_dict.get("metrics", {})
    elements_for_display = []
    for e in obs_dict.get("elements", []):
        elements_for_display.append({
            "id": e["id"],
            "type": e["type"],
            "cx": e["cx"],
            "cy": e["cy"],
            "w": e["w"],
            "h": e["h"],
        })
    return {
        "step": obs_dict.get("step", 0),
        "max_steps": obs_dict.get("max_steps", 20),
        "elements_json": json.dumps(elements_for_display, indent=2),
        "overlap": m.get("overlap", "?"),
        "boundary": m.get("boundary", "?"),
        "alignment": m.get("alignment", "?"),
        "spacing": m.get("spacing", "?"),
        "plausibility": m.get("plausibility", "?"),
        "quality_score": obs_dict.get("quality_score", "?"),
        "text_feedback": obs_dict.get("text_feedback", ""),
    }


def format_user_prompt(obs_dict: Dict) -> str:
    """Build the user-turn string from an observation dict."""
    return USER_PROMPT_TEMPLATE.format(**_metrics_format_args(obs_dict))


def format_vlm_user_prompt(obs_dict: Dict) -> List[Dict[str, Any]]:
    """
    Build OpenAI-style multimodal user content: text plus inline PNG (data URL).

    If ``rendered_image_base64`` is missing or empty the image block is omitted
    so the request stays valid (text-only fallback).
    """
    b64 = obs_dict.get("rendered_image_base64") or ""
    text = VLM_USER_PROMPT_TEMPLATE.format(**_metrics_format_args(obs_dict))
    parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    if b64:
        parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
        })
    return parts


ACTION_JSON_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "layout_action",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["MOVE", "RESIZE", "ALIGN", "SNAP", "NO_OP"],
                },
                "element_id": {"type": "integer"},
                "param": {
                    "type": "string",
                    "enum": [
                        "UP", "DOWN", "LEFT", "RIGHT",
                        "WIDER", "NARROWER", "TALLER", "SHORTER",
                        "CENTER_X", "CENTER_Y", "TOP", "BOTTOM",
                        "GRID", "NONE",
                    ],
                },
                "magnitude": {
                    "type": "string",
                    "enum": ["SMALL", "MEDIUM", "LARGE"],
                },
            },
            "required": ["action", "element_id", "param", "magnitude"],
            "additionalProperties": False,
        },
    },
}


def get_prompts(mode: str) -> Tuple[str, Callable[[Dict], Any]]:
    if mode == "vlm":
        return VLM_SYSTEM_PROMPT, format_vlm_user_prompt
    return SYSTEM_PROMPT, format_user_prompt


def parse_action(raw: str) -> Optional[LayoutAction]:
    """
    Try to parse the LM's raw text output into a LayoutAction.

    Handles ``<think>…</think>`` reasoning blocks and
    markdown code fences.  Returns None on any parse failure.
    """
    try:
        text = raw.strip()
        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        data = json.loads(text)

        magnitude = data.get("magnitude", "MEDIUM")
        if magnitude not in MAGNITUDES:
            magnitude = "MEDIUM"

        return LayoutAction(
            element_id=int(data["element_id"]),
            action=str(data["action"]).upper(),
            param=str(data["param"]).upper(),
            magnitude=magnitude,
        )
    except Exception:
        return None
