"""
Layout Environment Implementation.

An RL environment for iteratively refining UI poster layouts.
The agent receives a layout and must improve it using discrete actions
(MOVE, RESIZE, ALIGN, SNAP, NO_OP).

Perturbations are the responsibility of the caller (e.g. inference.py);
this environment is agnostic to how the initial layout was produced.
"""
from __future__ import annotations
import base64
import copy
import io
import json
import math
from pathlib import Path
import random
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import ACTIONS, MAGNITUDES, LayoutAction, LayoutObservation, LayoutState
except (ImportError, ModuleNotFoundError):
    from models import ACTIONS, MAGNITUDES, LayoutAction, LayoutObservation, LayoutState

from .metrics import (
    _axis_value,
    _to_ltrb,
    compute_all_metrics,
    quality_score,
)


# Single baked-in training-free layout (normalised bboxes).
# Training code should load the full dataset and pass ``sample=`` into ``reset``.
DEFAULT_LAYOUT_SAMPLE: Dict[str, Any] = {
    "id": 0,
    "canvas_size": [3556, 2000],
    "elements": [
        {
            "type": "Title",
            "text": "Demo",
            "bbox": [0.2, 0.15, 0.8, 0.25],
            "font_size": 120.0,
        },
        {
            "type": "Bodytext",
            "text": "Stateless default episode",
            "bbox": [0.15, 0.4, 0.85, 0.55],
            "font_size": 90.0,
        },
        {
            "type": "Website",
            "text": "example.com",
            "bbox": [0.35, 0.85, 0.65, 0.92],
            "font_size": 48.0,
        },
    ],
}


def _bbox_to_centre(bbox: List[float]) -> Dict[str, float]:
    x1, y1, x2, y2 = bbox
    return {
        "cx": (x1 + x2) / 2,
        "cy": (y1 + y2) / 2,
        "w": x2 - x1,
        "h": y2 - y1,
    }


def _default_stats_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Per-element-type Gaussian plausibility priors matching sample's ground truth.
    Shared isotropic covariance (loose prior) so perturbed layouts still score smoothly.
    """
    inv_cov = np.linalg.inv((0.1**2) * np.eye(5) + 1e-6 * np.eye(5))
    out: Dict[str, Any] = {}
    for elem in sample.get("elements", []):
        etype = elem.get("type")
        if not etype or etype in out:
            continue
        centre = _bbox_to_centre(elem["bbox"])
        canvas_h = float(sample["canvas_size"][1])
        fs_raw = float(elem.get("font_size", 0.0) or 0.0)
        fs_norm = fs_raw / canvas_h if canvas_h > 0 else 0.0
        mu = np.array(
            [centre["cx"], centre["cy"], centre["w"], centre["h"], fs_norm],
            dtype=np.float64,
        )
        out[etype] = {"mu": mu, "cov_inv": inv_cov}
    return out


DEFAULT_STATS: Dict[str, Any] = _default_stats_from_sample(DEFAULT_LAYOUT_SAMPLE)
SERVER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SERVER_DIR.parent
TASK_SAMPLES_PATH = REPO_ROOT / "dataset" / "task_samples.json"
DEFAULT_DATASET_JSON_PATH = str(REPO_ROOT / "dataset" / "genposter_5000_images.json")


def load_task_samples(path: Path = TASK_SAMPLES_PATH) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    return {entry["task_id"]: entry for entry in entries}


TASK_SAMPLE_MAP = load_task_samples()


def _perturb_sample(
    sample: Dict[str, Any],
    noise: float,
    *,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    perturbed = copy.deepcopy(sample)
    for elem in perturbed.get("elements", []):
        x1, y1, x2, y2 = elem["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        cx += rng.uniform(-noise, noise)
        cy += rng.uniform(-noise, noise)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.01, min(1.0, w))
        h = max(0.01, min(1.0, h))
        elem["bbox"] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return perturbed


def _sample_to_elements(sample: Dict) -> List[Dict]:
    """Convert a dataset sample to the internal element list."""
    canvas_w, canvas_h = sample["canvas_size"]
    elements = []
    for i, elem in enumerate(sample.get("elements", [])):
        centre = _bbox_to_centre(elem["bbox"])
        fs_raw = float(elem.get("font_size", 0.0) or 0.0)
        fs_norm = fs_raw / canvas_h if canvas_h > 0 else 0.0
        elements.append({
            "id": i,
            "type": elem.get("type", "unknown"),
            "text": elem.get("text", ""),
            "cx": centre["cx"],
            "cy": centre["cy"],
            "w": centre["w"],
            "h": centre["h"],
            "font_size": fs_norm,
        })
    return elements


# Action application
def _apply_action(
    elements: List[Dict],
    action: LayoutAction,
) -> None:
    """Mutate elements in-place according to action"""
    eid = action.element_id
    act = action.action
    param = action.param
    delta = MAGNITUDES.get(action.magnitude, MAGNITUDES["MEDIUM"])
    elem = elements[eid]

    if act == "MOVE":
        if param == "UP":
            elem["cy"] -= delta
        elif param == "DOWN":
            elem["cy"] += delta
        elif param == "LEFT":
            elem["cx"] -= delta
        elif param == "RIGHT":
            elem["cx"] += delta

    elif act == "RESIZE":
        if param == "WIDER":
            elem["w"] += delta
        elif param == "NARROWER":
            elem["w"] -= delta
        elif param == "TALLER":
            elem["h"] += delta
        elif param == "SHORTER":
            elem["h"] -= delta
        # Keep geometry valid for downstream metric computations.
        elem["w"] = max(0.01, min(1.0, elem["w"]))
        elem["h"] = max(0.01, min(1.0, elem["h"]))

    elif act == "ALIGN":
        _apply_align(elements, eid, param)

    elif act == "SNAP":
        grid = 0.05
        elem["cx"] = round(elem["cx"] / grid) * grid
        elem["cy"] = round(elem["cy"] / grid) * grid


_PARAM_TO_AXIS = {
    "LEFT": "left", "RIGHT": "right", "CENTER_X": "cx",
    "TOP": "top", "BOTTOM": "bottom", "CENTER_Y": "cy",
}


def _apply_align(
    elements: List[Dict],
    eid: int,
    param: str,
    threshold: float = 0.15,
) -> None:
    """Nearest-neighbour inter-element alignment with canvas fallback."""
    target = elements[eid]
    others = [e for e in elements if e["id"] != target["id"]]
    axis = _PARAM_TO_AXIS.get(param, param.lower())

    target_val = _axis_value(target, axis)
    best_val: Optional[float] = None
    best_dist = float("inf")

    for other in others:
        other_val = _axis_value(other, axis)
        dist = abs(target_val - other_val)
        if dist < best_dist:
            best_dist = dist
            best_val = other_val

    if best_val is not None and best_dist < threshold:
        snap_to = best_val
    else:
        canvas_anchors = {
            "left": 0.0, "right": 1.0, "cx": 0.5,
            "top": 0.0, "bottom": 1.0, "cy": 0.5,
        }
        snap_to = canvas_anchors[axis]

    _set_axis_value(target, axis, snap_to)


def _set_axis_value(e: Dict, axis: str, val: float) -> None:
    hw, hh = e["w"] / 2, e["h"] / 2
    if axis == "left":
        e["cx"] = val + hw
    elif axis == "right":
        e["cx"] = val - hw
    elif axis == "cx":
        e["cx"] = val
    elif axis == "top":
        e["cy"] = val + hh
    elif axis == "bottom":
        e["cy"] = val - hh
    elif axis == "cy":
        e["cy"] = val



# Round helpers
def _round_elements(elements: List[Dict], dp: int = 3) -> List[Dict]:
    """Return a copy with floats rounded for observation output."""
    out = []
    for e in elements:
        out.append({
            "id": e["id"],
            "type": e["type"],
            "cx": round(e["cx"], dp),
            "cy": round(e["cy"], dp),
            "w": round(e["w"], dp),
            "h": round(e["h"], dp),
            "font_size": round(e["font_size"], dp),
        })
    return out


def _resolve_media_path(dataset_json_path: str, relative_path: str) -> Path:
    """
    Resolve e.g. images/0_bg.png relative to the dataset JSON directory.
    This supports volume-mounted datasets when the server container can access
    the dataset path.
    """
    return Path(dataset_json_path).resolve().parent / relative_path


def _render_layout_on_background(
    bg_path: str | Path | None,
    elements: List[Dict],
    bg_pil: Image.Image | None = None,
) -> Image.Image:
    """
    Draw normalized layout (cx, cy, w, h in [0, 1]) on top of the background.
    Filled rectangles use distinct colors; label = type and truncated text.
    If bg_path is None or missing, a neutral placeholder canvas is used.
    bg_pil allows passing an already-loaded PIL image (e.g. decoded from
    base64) so the environment can work without filesystem access.
    """
    if bg_pil is not None:
        base = bg_pil.convert("RGBA")
        w_px, h_px = base.size
    elif bg_path is None:
        w_px, h_px = 1024, 1024
        base = Image.new("RGBA", (w_px, h_px), (245, 245, 245, 255))
    else:
        path = Path(bg_path)
        if not path.is_file():
            w_px, h_px = 1024, 1024
            base = Image.new("RGBA", (w_px, h_px), (245, 245, 245, 255))
        else:
            with Image.open(path) as img:
                base = img.convert("RGBA")
            w_px, h_px = base.size

    overlay = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    palette = [
        (255, 99, 71, 90),
        (60, 179, 113, 90),
        (65, 105, 225, 90),
        (238, 130, 238, 90),
        (255, 215, 0, 90),
        (0, 206, 209, 90),
        (255, 140, 0, 90),
        (147, 112, 219, 90),
    ]
    line_w = max(1, min(w_px, h_px) // 100)

    for i, e in enumerate(elements):
        cx, cy, ew, eh = (
            float(e["cx"]),
            float(e["cy"]),
            float(e["w"]),
            float(e["h"]),
        )
        x1 = int((cx - ew / 2) * w_px)
        y1 = int((cy - eh / 2) * h_px)
        x2 = int((cx + ew / 2) * w_px)
        y2 = int((cy + eh / 2) * h_px)
        x1 = max(0, min(x1, w_px - 1))
        y1 = max(0, min(y1, h_px - 1))
        x2 = max(0, min(x2, w_px - 1))
        y2 = max(0, min(y2, h_px - 1))
        if x2 <= x1:
            x2 = min(w_px - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h_px - 1, y1 + 1)

        fill = palette[i % len(palette)]
        outline = (*fill[:3], 255)
        draw.rectangle([x1, y1, x2, y2], fill=fill, outline=outline, width=line_w)

    composed = Image.alpha_composite(base, overlay)
    d2 = ImageDraw.Draw(composed)

    font_size = max(8, min(w_px, h_px) // 18)
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
        )
    except OSError:
        try:
            font = ImageFont.truetype("DejaVuSans.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()

    for i, e in enumerate(elements):
        cx, cy, ew, eh = (
            float(e["cx"]),
            float(e["cy"]),
            float(e["w"]),
            float(e["h"]),
        )
        x1 = int((cx - ew / 2) * w_px)
        y1 = int((cy - eh / 2) * h_px)
        x2 = int((cx + ew / 2) * w_px)
        y2 = int((cy + eh / 2) * h_px)
        x1 = max(0, min(x1, w_px - 1))
        y1 = max(0, min(y1, h_px - 1))
        x2 = max(0, min(x2, w_px - 1))
        y2 = max(0, min(y2, h_px - 1))
        if x2 <= x1:
            x2 = min(w_px - 1, x1 + 1)
        if y2 <= y1:
            y2 = min(h_px - 1, y1 + 1)

        raw_text = str(e.get("text", "") or "").strip()
        label = str(e.get("type", "unknown") or "unknown")
        if raw_text:
            label = f"{label}: {raw_text}"
        if len(label) > 48:
            label = label[:45] + "..."

        tb = d2.textbbox((0, 0), label, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        tx = x1 + max(2, (x2 - x1 - tw) // 2)
        ty = y1 + max(2, (y2 - y1 - th) // 2)

        d2.text(
            (tx, ty),
            label,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=max(1, line_w // 2),
            stroke_fill=(0, 0, 0, 255),
        )

    if composed.mode == "RGBA":
        rgb = Image.new("RGB", composed.size, (255, 255, 255))
        rgb.paste(composed, mask=composed.split()[3])
        return rgb
    return composed.convert("RGB")


def _pil_to_png_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _generate_text_feedback(
    delta_q: float,
    metrics: Dict[str, float],
    elements: List[Dict],
) -> str:
    """
    Produce a concise, actionable text hint from the current metrics.

    The feedback tells the model (a) whether it improved, and (b) which
    metric to target next with a concrete suggestion.
    """
    parts: List[str] = []

    if delta_q > 0.01:
        parts.append(f"Quality improved by +{delta_q:.3f}. Keep going.")
    elif delta_q < -0.01:
        parts.append(f"Quality dropped by {delta_q:.3f}. Undo or try a different action.")
    else:
        parts.append("Negligible change. Try a different element or direction.")

    overlap = metrics.get("overlap", 0.0)
    boundary = metrics.get("boundary", 0.0)
    alignment = metrics.get("alignment", 1.0)
    spacing = metrics.get("spacing", 1.0)

    penalties = {"overlap": overlap, "boundary": boundary}
    worst_penalty_name = max(penalties, key=penalties.get)  # type: ignore[arg-type]
    worst_penalty_val = penalties[worst_penalty_name]

    rewards = {"alignment": alignment, "spacing": spacing}
    worst_reward_name = min(rewards, key=rewards.get)  # type: ignore[arg-type]
    worst_reward_val = rewards[worst_reward_name]

    if worst_penalty_val > 0.05:
        if worst_penalty_name == "overlap":
            parts.append(
                f"Overlap is high ({overlap:.3f}). "
                "MOVE overlapping elements apart or RESIZE them smaller."
            )
        else:
            oob = [
                e["id"] for e in elements
                if _is_out_of_bounds(e)
            ]
            if oob:
                parts.append(
                    f"Boundary violation ({boundary:.3f}) on element(s) {oob}. "
                    "MOVE them inward or RESIZE them smaller."
                )
            else:
                parts.append(
                    f"Boundary penalty ({boundary:.3f}). "
                    "Some elements may be near the edge; MOVE inward."
                )
    elif worst_reward_val < 0.5:
        if worst_reward_name == "alignment":
            parts.append(
                f"Alignment is low ({alignment:.3f}). "
                "Use ALIGN (CENTER_X, LEFT, etc.) to snap edges together."
            )
        else:
            parts.append(
                f"Spacing is uneven ({spacing:.3f}). "
                "MOVE elements to equalise vertical/horizontal gaps."
            )

    return " ".join(parts)


def _is_out_of_bounds(e: Dict) -> bool:
    hw, hh = e["w"] / 2, e["h"] / 2
    l, t, r, b = e["cx"] - hw, e["cy"] - hh, e["cx"] + hw, e["cy"] + hh
    return l < 0 or t < 0 or r > 1 or b > 1


# Environment
INVALID_ACTION_PENALTY = -0.5
STEP_PENALTY = -0.05      
REWARD_SCALE = 10.0
TERMINAL_BONUS_SCALE = 5.0
TERMINAL_PENALTY = -1.0
# Align terminal shaping with the easiest grader delta threshold.
Q_DELTA_THRESHOLD = 0.05
VISIBLE_REWARD_EPS = 0.01


def _normalize_visible_reward(raw_reward: float | int) -> float:
    """
    Bound the externally visible reward to the open interval (0, 1).

    The environment still computes its internal shaping reward in the native
    scale, but the reward exposed through the OpenEnv API should remain
    validator-friendly and render safely at 2 decimal places.
    """
    normalized = 1.0 / (1.0 + math.exp(-float(raw_reward)))
    return min(max(normalized, VISIBLE_REWARD_EPS), 1.0 - VISIBLE_REWARD_EPS)


class LayoutEnvironment(Environment):
    """
    An RL environment for layout refinement.

    The caller is responsible for producing the initial layout (e.g. by
    perturbing a ground-truth sample) and passing it via reset(sample=...).

    Args:
        max_steps: Maximum actions per episode.
        weights: Optional metric weight overrides for Q.
        stats: Plausibility metric config (e.g. loaded from *_stats.npy);
               immutable for the lifetime of this env instance. If omitted,
               DEFAULT_STATS (derived from DEFAULT_LAYOUT_SAMPLE) is used.
    """

    # This environment stores episode-specific fields on the instance.
    # Do not advertise shared-instance concurrent session safety.
    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    def __init__(
        self,
        max_steps: int = 500,
        weights: Optional[Dict[str, float]] = None,
        stats: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self._state = LayoutState(episode_id=str(uuid4()), step_count=0)

        self._max_steps = max_steps
        self._active_max_steps = max_steps
        self._weights = weights
        self._stats: Dict[str, Any] = (
            DEFAULT_STATS if stats is None else stats
        )
        self._mode: Literal["llm", "vlm"] = "llm"
        self._text_feedback: bool = True
        self._render_image_in_observation: bool = True
        self._task_id: str = "default"

    def _build_observation(
        self,
        step_num: int,
        done: bool,
        reward: float | int,
        metrics: Dict,
        q: float,
    ) -> LayoutObservation:
        image_path: Optional[str] = None
        rendered_b64: Optional[str] = None
        if self._mode == "vlm":
            image_path = self._state.current_image_rel
        if self._mode == "vlm" and self._render_image_in_observation:
            resolved_bg_path: Optional[Path] = None
            bg_img: Image.Image | None = None

            inline_b64 = getattr(self._state, "_bg_image_base64", None)
            if inline_b64:
                with Image.open(io.BytesIO(base64.b64decode(inline_b64))) as decoded:
                    bg_img = decoded.convert("RGBA")
            elif self._state.current_image_rel and self._state.dataset_json_path:
                resolved_bg_path = _resolve_media_path(
                    self._state.dataset_json_path, self._state.current_image_rel
                )
                if resolved_bg_path.is_file():
                    with Image.open(resolved_bg_path) as loaded:
                        bg_img = loaded.convert("RGBA")

            rendered = _render_layout_on_background(
                resolved_bg_path, self._state.elements, bg_pil=bg_img
            )
            rendered_b64 = _pil_to_png_base64(rendered)

        prev_q = self._state.previous_quality
        delta_q = q - prev_q

        feedback: Optional[str] = None
        if self._text_feedback:
            if step_num == 0:
                feedback = "Episode started. Choose an element and action."
            else:
                feedback = _generate_text_feedback(delta_q, metrics, self._state.elements)

        obs = LayoutObservation(
            canvas={"width": 1.0, "height": 1.0},
            elements=_round_elements(self._state.elements),
            metrics=metrics,
            step=step_num,
            max_steps=self._active_max_steps,
            quality_score=q,
            initial_quality_score=self._state.initial_quality,
            text_feedback=feedback,
            reward=_normalize_visible_reward(reward),
            done=done,
            metadata={"task_id": self._task_id},
            image_path=image_path,
            rendered_image_base64=rendered_b64,
        )
        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        *,
        task_id: Optional[str] = None,
        sample: Optional[Dict[str, Any]] = None,
        dataset_json_path: Optional[str] = None,
        background_image_base64: Optional[str] = None,
        mode: Optional[Literal["llm", "vlm"]] = None,
        text_feedback: Optional[bool] = None,
        render_image_in_observation: Optional[bool] = None,
        **kwargs: Any,
    ) -> LayoutObservation:
        # Intentionally avoid touching module-global RNG state here.
        # Seeding happens client-side for perturbation reproducibility.

        if mode is not None:
            self._mode = mode
        if text_feedback is not None:
            self._text_feedback = text_feedback
        if render_image_in_observation is not None:
            self._render_image_in_observation = render_image_in_observation

        episode_max_steps = self._max_steps
        selected_task_id = task_id or "default"
        if sample is not None:
            chosen = sample
        elif task_id is not None:
            if task_id not in TASK_SAMPLE_MAP:
                raise ValueError(
                    f"Unknown task_id '{task_id}'. Expected one of {sorted(TASK_SAMPLE_MAP)}"
                )
            task_entry = TASK_SAMPLE_MAP[task_id]
            chosen = _perturb_sample(
                task_entry["sample"],
                float(task_entry.get("noise", 0.0)),
                seed=seed,
            )
            episode_max_steps = int(task_entry.get("max_steps", self._max_steps))
        else:
            chosen = DEFAULT_LAYOUT_SAMPLE

        self._active_max_steps = episode_max_steps
        self._task_id = selected_task_id

        if dataset_json_path is None:
            dataset_json_path = DEFAULT_DATASET_JSON_PATH

        if self._mode == "vlm" and not chosen.get("image_path") and not background_image_base64:
            raise ValueError(
                "VLM mode requires sample['image_path'] or background_image_base64. "
                "Pass a sample from your dataset on reset."
            )

        current_image_rel = (
            chosen.get("image_path") if self._mode == "vlm" else None
        )

        elements = _sample_to_elements(chosen)

        self._state = LayoutState(
            episode_id=episode_id if episode_id is not None else str(uuid4()),
            step_count=0,
            elements=elements,
            previous_quality=0.0,
            initial_quality=0.0,
            current_image_rel=current_image_rel,
            dataset_json_path=dataset_json_path,
        )

        if background_image_base64:
            self._state._bg_image_base64 = background_image_base64

        metrics = compute_all_metrics(self._state.elements, self._stats)
        q = quality_score(metrics, self._weights)
        self._state.previous_quality = q
        self._state.initial_quality = q

        return self._build_observation(0, False, 0.0, metrics, q)

    def step(self, action: LayoutAction) -> LayoutObservation:  # type: ignore[override]
        self._state.step_count += 1
        step_num = self._state.step_count

        valid = action.is_valid(len(self._state.elements))

        if not valid:
            metrics = compute_all_metrics(self._state.elements, self._stats)
            q = quality_score(metrics, self._weights)
            done = step_num >= self._active_max_steps
            reward = INVALID_ACTION_PENALTY + STEP_PENALTY
            if done:
                q_delta = q - self._state.initial_quality
                reward += (
                    TERMINAL_BONUS_SCALE if q_delta >= Q_DELTA_THRESHOLD else TERMINAL_PENALTY
                )
            return self._build_observation(
                step_num, done, round(reward, 4), metrics, q
            )

        is_noop = action.action == "NO_OP"

        if not is_noop:
            _apply_action(self._state.elements, action)
        else:
            pass

        metrics = compute_all_metrics(self._state.elements, self._stats)
        q = quality_score(metrics, self._weights)
        delta_q = q - self._state.previous_quality

        done = is_noop or step_num >= self._active_max_steps

        reward = REWARD_SCALE * delta_q + STEP_PENALTY
        if done:
            q_delta = q - self._state.initial_quality
            reward += (
                TERMINAL_BONUS_SCALE if q_delta >= Q_DELTA_THRESHOLD else TERMINAL_PENALTY
            )

        obs = self._build_observation(
            step_num, done, round(reward, 4), metrics, q
        )
        self._state.previous_quality = q
        return obs

    @property
    def state(self) -> LayoutState:
        return self._state
