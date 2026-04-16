"""
Gradio UI for the layout refinement environment.

Connects to a deployed LayoutEnv (e.g. Hugging Face Space), runs the same
LLM loop as inference.py, and visualises ground-truth boxes on the left and
per-step refined layouts on the right.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import math
import os
import random
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image, ImageDraw

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"

try:
    from layoutenv import LayoutAction, LayoutEnv
    from layoutenv.client import layout_env_kwargs_from_environ, warmup_hf_space_http
except ImportError:
    from client import LayoutEnv, layout_env_kwargs_from_environ, warmup_hf_space_http
    from models import LayoutAction

from inference import (
    DATASET_JSON_SERVER,
    get_layout_action,
    load_task_samples,
    perturb_sample,
)

# Mirror inference.py env defaults
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
# Gradio URL field: prefer GRADIO_ENV_BASE_URL so a local ENV_BASE_URL for Docker inference
# does not override the Space URL in the UI. Default is the deployed HF Space.
_DEFAULT_HF_SPACE = "https://ryz3n758-layoutenv.hf.space"
ENV_BASE_URL = os.getenv("GRADIO_ENV_BASE_URL") or os.getenv("LAYOUTENV_BASE_URL") or _DEFAULT_HF_SPACE
# Pause between streamed yields so the browser can repaint each layout (Gradio may otherwise show only the last frame).
STEP_PAUSE_SEC = float(os.getenv("GRADIO_STEP_PAUSE_SEC", "0.45"))

# --- Colours: one distinct colour per element type (no text labels on boxes) ---

_TYPE_PALETTE: Dict[str, Tuple[int, int, int]] = {
    "Title": (220, 50, 47),
    "Bodytext": (38, 139, 210),
    "Website": (133, 153, 0),
    "Date": (211, 54, 130),
    "Calls to Action": (181, 137, 0),
    "Location": (42, 161, 152),
    "Social Media": (108, 113, 196),
    "Name": (203, 75, 22),
}


def _color_for_type(type_name: str) -> Tuple[int, int, int]:
    if type_name in _TYPE_PALETTE:
        return _TYPE_PALETTE[type_name]
    digest = hashlib.md5(type_name.encode("utf-8")).hexdigest()
    return (int(digest[0:2], 16), int(digest[2:4], 16), int(digest[4:6], 16))


def _upscale_if_tiny(img: Image.Image, min_side: int = 448) -> Image.Image:
    """Poster crops are often 224×224; upscale for clearer box overlays."""
    w, h = img.size
    m = max(w, h)
    if m >= min_side:
        return img.copy()
    scale = min_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)


def _ltrb_from_obs_element(e: Dict[str, Any]) -> Tuple[float, float, float, float]:
    cx, cy = float(e["cx"]), float(e["cy"])
    hw, hh = float(e["w"]) / 2.0, float(e["h"]) / 2.0
    return cx - hw, cy - hh, cx + hw, cy + hh


def _ltrb_from_gt_bbox(bbox: List[float]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return float(x1), float(y1), float(x2), float(y2)


def _draw_boxes_on_image(
    base: Image.Image,
    boxes: List[Tuple[Tuple[int, int, int], Tuple[float, float, float, float]]],
    line_width: int = 3,
) -> Image.Image:
    """boxes: list of (rgb, (x1,y1,x2,y2)) in normalised [0,1] coordinates."""
    img = base.convert("RGBA")
    draw = ImageDraw.Draw(img)
    w, h = img.size
    for color, (x1, y1, x2, y2) in boxes:
        px1, py1 = x1 * w, y1 * h
        px2, py2 = x2 * w, y2 * h
        draw.rectangle([px1, py1, px2, py2], outline=color + (255,), width=line_width)
    return img.convert("RGB")


def render_gt_panel(sample: Dict[str, Any]) -> Image.Image:
    """Ground-truth bounding boxes (dataset sample, unperturbed) on the background."""
    rel = sample.get("image_path")
    if not rel:
        raise ValueError("sample has no image_path")
    bg_path = DATASET_DIR / rel
    if not bg_path.is_file():
        raise FileNotFoundError(f"Background not found: {bg_path}")
    with Image.open(bg_path) as im:
        base = _upscale_if_tiny(im.convert("RGBA"))
    boxes: List[Tuple[Tuple[int, int, int], Tuple[float, float, float, float]]] = []
    for elem in sample.get("elements", []):
        t = str(elem.get("type", "unknown"))
        c = _color_for_type(t)
        boxes.append((c, _ltrb_from_gt_bbox(elem["bbox"])))
    return _draw_boxes_on_image(base, boxes)


def render_refinement_on_background(
    bg_rgba: Image.Image,
    elements: List[Dict[str, Any]],
) -> Image.Image:
    """Coloured boxes for observation elements on top of the same background image."""
    boxes: List[Tuple[Tuple[int, int, int], Tuple[float, float, float, float]]] = []
    for e in elements:
        t = str(e.get("type", "unknown"))
        c = _color_for_type(t)
        boxes.append((c, _ltrb_from_obs_element(e)))
    return _draw_boxes_on_image(bg_rgba, boxes)


def _load_background_for_task(sample: Dict[str, Any]) -> Image.Image:
    rel = sample.get("image_path")
    if not rel:
        raise ValueError("sample has no image_path")
    bg_path = DATASET_DIR / rel
    if not bg_path.is_file():
        raise FileNotFoundError(f"Background not found: {bg_path}")
    with Image.open(bg_path) as im:
        return _upscale_if_tiny(im.convert("RGBA"))


def _slider_update(n_frames: int, *, interactive: bool) -> gr.update:
    if n_frames <= 0:
        return gr.update(
            minimum=0,
            maximum=0,
            value=0,
            step=1,
            interactive=False,
            label="Replay step",
        )
    last = n_frames - 1
    return gr.update(
        minimum=0,
        maximum=last,
        value=last,
        step=1,
        interactive=interactive,
        label=f"Replay step (0–{last})",
    )


def _replay_frame(idx: float, frames: List[Image.Image]) -> Optional[Image.Image]:
    if not frames:
        return None
    i = int(round(float(idx)))
    i = max(0, min(i, len(frames) - 1))
    return frames[i]


async def run_visual_episode(
    task_id: str,
    env_base_url: str,
    seed: float,
    max_steps: int,
    mode: str,
    progress: gr.Progress = gr.Progress(),
) -> AsyncIterator[
    Tuple[
        Optional[Image.Image],
        Optional[Image.Image],
        str,
        str,
        str,
        Any,
        List[Image.Image],
    ]
]:
    """
    Async generator: yields after each layout snapshot so the UI can repaint.
    Stores full ``frames`` in State for the replay slider.
    """
    z_sl = gr.update(minimum=0, maximum=0, value=0, interactive=False)

    def pack(
        gt: Optional[Image.Image],
        refine: Optional[Image.Image],
        live: str,
        status: str,
        feedback: str,
        slider: Any,
        frames: List[Image.Image],
    ):
        return (gt, refine, live, status, feedback, slider, frames)

    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        yield pack(
            None,
            None,
            "",
            "Set **HF_TOKEN** in the environment (see `.env`) for the Hugging Face router.",
            "",
            z_sl,
            [],
        )
        return

    tasks = load_task_samples()
    task_entry = next((t for t in tasks if t["task_id"] == task_id), None)
    if not task_entry:
        yield pack(None, None, "", f"Unknown task: {task_id}", "", z_sl, [])
        return

    gt_sample = copy.deepcopy(task_entry["sample"])
    try:
        left = render_gt_panel(gt_sample)
    except (FileNotFoundError, ValueError) as exc:
        yield pack(None, None, "", f"Could not load ground-truth image: {exc}", "", z_sl, [])
        return

    # Show ground truth immediately (local render) before any remote I/O so the left panel
    # does not stay blank until the episode finishes.
    yield pack(
        left,
        None,
        "**Live:** connecting to remote env (left image stays fixed).",
        "**Connecting…**  Warming Space / WebSocket.",
        "_Environment feedback will appear after the remote reset completes._",
        z_sl,
        [],
    )

    noise = task_entry.get("noise", 0.1)
    cap = task_entry.get("max_steps", 100)
    if max_steps > 0:
        cap = min(cap, int(max_steps))
    if seed is not None and not (isinstance(seed, float) and math.isnan(seed)):
        random.seed(int(seed))

    perturbed = perturb_sample(gt_sample, noise=noise)
    client = OpenAI(base_url=API_BASE_URL, api_key=api_key)
    env: Optional[LayoutEnv] = None
    frames: List[Image.Image] = []
    feedback_sections: List[str] = []

    try:
        await asyncio.to_thread(warmup_hf_space_http, env_base_url)
        env = LayoutEnv(
            base_url=env_base_url.rstrip("/"),
            **layout_env_kwargs_from_environ(),
        )
    except Exception as exc:
        yield pack(
            left,
            None,
            f"**Live:** connection failed.",
            f"**Could not create env client:** `{exc}`",
            "",
            z_sl,
            [],
        )
        return

    yield pack(
        left,
        None,
        "**Live:** resetting remote episode…",
        "**Resetting remote episode…**",
        "_Waiting for first observation…_",
        z_sl,
        [],
    )

    status_lines: List[str] = []

    try:
        progress(0.0, desc="Resetting remote environment…")
        result = await env.reset(
            seed=int(seed) if seed is not None else None,
            sample=perturbed,
            dataset_json_path=DATASET_JSON_SERVER,
            mode=mode,
            render_image_in_observation=True,
        )
        obs = result.observation
        initial_q = obs.quality_score

        bg_rgba = _load_background_for_task(perturbed)
        step0_img = render_refinement_on_background(bg_rgba, obs.elements)
        last_refine = step0_img
        frames.append(step0_img)

        fb = (obs.text_feedback or "").strip()
        if fb:
            feedback_sections.append(f"##### Step 0 (initial)\n{fb}")

        await asyncio.sleep(STEP_PAUSE_SEC)
        yield pack(
            left,
            step0_img,
            f"**Live:** step **0** / {cap} (initial layout)  ·  Q={obs.quality_score:.3f}",
            f"**Current:** Step 0 (perturbed)  ·  Q={obs.quality_score:.3f}  ·  **{cap}** refinement steps planned",
            "\n\n".join(feedback_sections) if feedback_sections else "_No feedback yet._",
            _slider_update(len(frames), interactive=False),
            list(frames),
        )

        history: List[dict] = []
        pending_feedback: Optional[str] = None

        for step in range(1, cap + 1):
            if result.done:
                break
            progress(min(step / max(cap, 1), 0.99), desc=f"Step {step}/{cap}…")
            rendered = obs.rendered_image_base64 if mode == "vlm" else None
            action, raw, parse_failed, api_error = await get_layout_action(
                client,
                history,
                obs,
                mode,
                rendered_b64=rendered,
                pending_feedback=pending_feedback,
            )
            pending_feedback = None
            if action is None:
                action = LayoutAction(
                    element_id=0,
                    action="MOVE",
                    param="UP",
                    magnitude="SMALL",
                )
            if api_error:
                status_lines.append(f"Step {step}: API warning — {api_error}")
            result = await env.step(action)
            obs = result.observation
            last_refine = render_refinement_on_background(bg_rgba, obs.elements)
            frames.append(last_refine)

            fb = (obs.text_feedback or "").strip()
            if fb:
                feedback_sections.append(f"##### After refinement {step}\n{fb}")

            await asyncio.sleep(STEP_PAUSE_SEC)
            yield pack(
                left,
                last_refine,
                f"**Live:** refinement **{step}** / {cap}  ·  Q={obs.quality_score:.3f}",
                f"**Current:** Refinement **{step}** / {cap}  ·  Q={obs.quality_score:.3f}  ·  Q₀={initial_q:.3f}",
                "\n\n".join(feedback_sections),
                _slider_update(len(frames), interactive=False),
                list(frames),
            )

            if obs.text_feedback:
                pending_feedback = obs.text_feedback

            if result.done:
                break

        final_q = obs.quality_score
        summary = (
            f"**Task:** `{task_id}`  ·  **Initial Q:** {initial_q:.3f}  ·  **Final Q:** {final_q:.3f}  ·  "
            f"**Layouts captured:** {len(frames)} (step 0 + {len(frames) - 1} refinements)\n\n"
            + ("\n\n".join(status_lines) if status_lines else "")
        )
        yield pack(
            left,
            last_refine,
            f"**Live:** finished — **{len(frames)}** layouts (step 0 + {len(frames) - 1} refinements).",
            summary.strip(),
            "\n\n".join(feedback_sections) if feedback_sections else "_No text feedback._",
            _slider_update(len(frames), interactive=True),
            list(frames),
        )
    except Exception as exc:
        yield pack(
            left,
            None,
            "**Live:** error — see Run summary.",
            f"**Error:** `{exc}`",
            "\n\n".join(feedback_sections) if feedback_sections else "",
            z_sl,
            list(frames),
        )
    finally:
        if env is not None:
            try:
                await env.close()
            except Exception:
                pass


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="Layout refinement viewer") as demo:
        gr.Markdown(
            "## Layout refinement\n"
            "Left: **ground-truth** boxes on the poster. "
            "Right: **live refinement**. "
            "Use the **replay** slider after the run to scrub through layouts. "
            "**Environment feedback** from the server is appended after each step. "
            "Episodes run for the **max steps** you set.\n\n"
            f"*Stream pacing: `{STEP_PAUSE_SEC}` s between frames (set `GRADIO_STEP_PAUSE_SEC`).*"
        )
        with gr.Row():
            gt_out = gr.Image(label="Ground truth", type="pil", height=480)
            refine_out = gr.Image(
                label="Refinement",
                type="pil",
                height=480,
            )
        gr.Markdown("### Live run")
        live_md = gr.Markdown(
            value="_Progress appears here without covering the ground-truth image._",
        )
        with gr.Row():
            step_slider = gr.Slider(
                minimum=0,
                maximum=0,
                value=0,
                step=1,
                label="Replay step",
                interactive=False,
            )
        with gr.Accordion("Run summary", open=False):
            status = gr.Markdown()
        with gr.Accordion("Environment feedback (per step)", open=False):
            feedback_md = gr.Markdown(
                value="_Feedback from the remote env appears here after each step._",
            )
        frames_state = gr.State([])
        with gr.Row():
            task_dd = gr.Dropdown(
                choices=["easy", "medium", "hard"],
                value="easy",
                label="Task",
            )
            mode_dd = gr.Dropdown(
                choices=["vlm", "llm"],
                value="vlm",
                label="Mode",
            )
            max_steps_sl = gr.Slider(
                minimum=1,
                maximum=100,
                value=5,
                step=1,
                label="Max steps",
            )
            seed_num = gr.Number(value=42, label="Seed", precision=0)
        env_url = gr.Textbox(
            label="Layout env base URL",
            value=ENV_BASE_URL,
            placeholder="https://your-space.hf.space",
        )
        run_btn = gr.Button("Run episode", variant="primary")

        run_btn.click(
            fn=run_visual_episode,
            inputs=[task_dd, env_url, seed_num, max_steps_sl, mode_dd],
            outputs=[
                gt_out,
                refine_out,
                live_md,
                status,
                feedback_md,
                step_slider,
                frames_state,
            ],
            # "full" draws the step progress bar *inside the first output* (ground-truth Image),
            # which replaces the image. Use hidden + `live_md` for step text instead.
            show_progress="hidden",
        )

        step_slider.change(
            fn=_replay_frame,
            inputs=[step_slider, frames_state],
            outputs=[refine_out],
        )

    return demo


def main() -> None:
    demo = build_demo()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
