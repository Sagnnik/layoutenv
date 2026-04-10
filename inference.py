"""
Inference Script — Layout RL Environment
===================================
STDOUT FORMAT
- The script emits exactly:
    [START] ...
    [STEP] ...
    [END] ...
"""

import argparse
import asyncio
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI
try:
    from layoutenv import LayoutAction, LayoutEnv, LayoutObservation
    from layoutenv.grader import grade_episode, success_from_q_delta
except ImportError:
    from client import LayoutEnv
    from grader import grade_episode, success_from_q_delta
    from models import LayoutAction, LayoutObservation
from prompts import ACTION_JSON_SCHEMA, get_prompts, parse_action

load_dotenv()

IMAGE_NAME = os.getenv("IMAGE_NAME", "layoutenv:latest")
ENV_BASE_URL = os.getenv("LAYOUTENV_BASE_URL") or os.getenv("ENV_BASE_URL", "https://ryz3n758-layoutenv.hf.space")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
TASK_NAME = os.getenv("LAYOUT_TASK", "layout-refinement")
BENCHMARK = os.getenv("LAYOUT_BENCHMARK", "layoutenv")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = 200
SUCCESS_Q_DELTA = 0.1
PRINT_SUMMARY_STDERR = os.getenv("PRINT_SUMMARY_STDERR", "0") == "1"
EARLY_STOP_ON_SUCCESS = os.getenv("EARLY_STOP_ON_SUCCESS", "1") == "1"

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset"
TASK_SAMPLES_JSON = DATASET_DIR / "task_samples.json"
DATASET_JSON_LOCAL = os.getenv("LAYOUT_DATASET", str(DATASET_DIR / "genposter_5000_images.json"))
DATASET_JSON_SERVER = os.getenv("LAYOUT_DATASET_SERVER", "/app/env/dataset/genposter_5000_images.json")
USE_STRUCTURED_OUTPUT = os.getenv("USE_STRUCTURED_OUTPUT", "0") == "1"
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "12"))


def load_task_samples(path: str | Path = TASK_SAMPLES_JSON) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def perturb_sample(sample: Dict[str, Any], noise: float = 0.1) -> Dict[str, Any]:
    perturbed = copy.deepcopy(sample)
    for elem in perturbed.get("elements", []):
        x1, y1, x2, y2 = elem["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        cx += random.uniform(-noise, noise)
        cy += random.uniform(-noise, noise)
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.01, min(1.0, w))
        h = max(0.01, min(1.0, h))
        elem["bbox"] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
    return perturbed


def resolve_background_image_path(sample: Dict[str, Any], dataset_json_path: str) -> Optional[Path]:
    rel = sample.get("image_path")
    if not rel:
        return None
    abs_path = Path(dataset_json_path).resolve().parent / rel
    if not abs_path.is_file():
        print(f"[WARN] background image not found: {abs_path}", file=sys.stderr)
        return None
    return abs_path


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action_str: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def obs_to_dict(obs: LayoutObservation, rendered_b64: Optional[str] = None) -> dict:
    return {
        "canvas": obs.canvas,
        "elements": obs.elements,
        "metrics": obs.metrics,
        "step": obs.step,
        "max_steps": obs.max_steps,
        "quality_score": obs.quality_score,
        "text_feedback": obs.text_feedback or "",
        "rendered_image_base64": rendered_b64,
    }


async def get_layout_action(
    client: OpenAI,
    history: List[dict],
    obs: LayoutObservation,
    mode: str = "llm",
    rendered_b64: Optional[str] = None,
    pending_feedback: Optional[str] = None,
) -> tuple[Optional[LayoutAction], str, bool]:
    system_prompt, format_user_msg = get_prompts(mode)
    obs_payload = obs_to_dict(obs, rendered_b64=rendered_b64)
    if pending_feedback:
        obs_payload["text_feedback"] = ((obs_payload.get("text_feedback") or "") + "\n" + pending_feedback).strip()
    user_content = format_user_msg(obs_payload)
    messages = [{"role": "system", "content": system_prompt}] + history
    messages.append({"role": "user", "content": user_content})
    api_kwargs: Dict[str, Any] = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    if USE_STRUCTURED_OUTPUT:
        api_kwargs["response_format"] = ACTION_JSON_SCHEMA
    try:
        completion = await asyncio.to_thread(client.chat.completions.create, **api_kwargs)
        raw = (completion.choices[0].message.content or "").strip()
    except Exception:
        raw = ""
    action = parse_action(raw)
    parse_failed = action is None
    history.append({"role": "user", "content": user_content})
    history.append({"role": "assistant", "content": raw if raw else "{}"})
    # Keep only the most recent N turns (2 messages per turn).
    keep_messages = max(0, MAX_HISTORY_TURNS * 2)
    if keep_messages and len(history) > keep_messages:
        del history[:-keep_messages]
    return action, raw, parse_failed


def action_to_string(action: Optional[LayoutAction], raw: str) -> str:
    if action is None:
        return f"PARSE_FAIL({raw[:50]})"
    s = f"{action.action}(eid={action.element_id},p={action.param}"
    if action.action in ("MOVE", "RESIZE"):
        s += f",mag={action.magnitude}"
    return s + ")"


async def run_episode(
    client: OpenAI,
    env: LayoutEnv,
    task_entry: Dict[str, Any],
    mode: str,
    seed: Optional[int],
    *,
    dataset_json_server: str,
    max_steps_override: Optional[int] = None,
) -> Dict[str, Any]:
    task_id = task_entry["task_id"]
    sample = task_entry["sample"]
    noise = task_entry.get("noise", 0.1)
    max_steps = task_entry.get("max_steps", 100)
    if max_steps_override is not None and max_steps_override > 0:
        max_steps = min(max_steps, max_steps_override)
    if seed is not None:
        random.seed(seed)
    perturbed = perturb_sample(sample, noise=noise)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    summary: Dict[str, Any] = {
        "task_id": task_id,
        "success": False,
        "steps": 0,
        "score": 0.0,
        "initial_q": 0.0,
        "final_q": 0.0,
        "q_delta": 0.0,
    }
    run_error: Optional[Exception] = None
    try:
        result = await env.reset(
            seed=seed,
            sample=perturbed,
            dataset_json_path=dataset_json_server,
            mode=mode,
            render_image_in_observation=True,
        )
        obs = result.observation
        initial_q = obs.quality_score
        history: List[dict] = []
        pending_feedback: Optional[str] = None
        for step in range(1, max_steps + 1):
            if result.done:
                break
            rendered = obs.rendered_image_base64 if mode == "vlm" else None
            action, raw, _parse_failed = await get_layout_action(
                client,
                history,
                obs,
                mode,
                rendered_b64=rendered,
                pending_feedback=pending_feedback,
            )
            pending_feedback = None
            if action is None:
                action = LayoutAction(element_id=0, action="NO_OP", param="NONE")
            action_str = action_to_string(action, raw)
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            raw_error = getattr(result, "last_action_error", None)
            if raw_error is None:
                metadata = getattr(obs, "metadata", None)
                if isinstance(metadata, dict):
                    raw_error = metadata.get("last_action_error")
            if obs.text_feedback:
                # Inject as part of next user payload to avoid consecutive user-role messages.
                pending_feedback = obs.text_feedback
            rewards.append(reward)
            steps_taken = step
            log_step(step, action_str, reward, done, raw_error)
            if EARLY_STOP_ON_SUCCESS:
                q_delta_now = obs.quality_score - initial_q
                if success_from_q_delta(task_id, q_delta_now, SUCCESS_Q_DELTA):
                    break
            if done:
                break
        final_q = obs.quality_score
        q_delta = final_q - initial_q
        grade = grade_episode(
            task_id=task_id,
            initial_quality=initial_q,
            final_quality=final_q,
            success_q_delta=SUCCESS_Q_DELTA,
        )
        success = grade.success
        summary = {
            "task_id": task_id,
            "success": grade.success,
            "steps": steps_taken,
            "score": grade.score,
            "initial_q": initial_q,
            "final_q": final_q,
            "q_delta": q_delta,
        }
    except Exception as exc:
        run_error = exc
    finally:
        try:
            await env.close()
        except Exception as close_error:
            print(f"[DEBUG] env.close() error (container cleanup): {close_error}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, rewards=rewards)
    if run_error is not None:
        raise run_error
    return summary


async def main() -> None:
    parser = argparse.ArgumentParser(description="Layout RL inference script")
    parser.add_argument("--mode", choices=["llm", "vlm"], default="llm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default=None, choices=["easy", "medium", "hard"])
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument(
        "--env-base-url",
        type=str,
        default=ENV_BASE_URL,
        help="Remote environment base URL (e.g. https://<space>.hf.space). "
        "If set, skip local Docker startup.",
    )
    args = parser.parse_args()
    if not HF_TOKEN:
        raise RuntimeError("Missing required HF_TOKEN environment variable.")
    tasks = load_task_samples()
    if args.task:
        tasks = [t for t in tasks if t["task_id"] == args.task]
        if not tasks:
            raise RuntimeError(f"Task '{args.task}' not found in {TASK_SAMPLES_JSON}")
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    results: List[Dict[str, Any]] = []
    for task_entry in tasks:
        if args.env_base_url:
            env = LayoutEnv(base_url=args.env_base_url.rstrip("/"))
        else:
            env = await LayoutEnv.from_docker_image(IMAGE_NAME)
        try:
            r = await run_episode(
                client,
                env,
                task_entry,
                args.mode,
                args.seed,
                dataset_json_server=DATASET_JSON_SERVER,
                max_steps_override=args.max_steps,
            )
            results.append(r)
        except Exception:
            # run_episode always emits [END], then re-raises.
            raise
    if PRINT_SUMMARY_STDERR:
        print("\n=== Summary ===", file=sys.stderr, flush=True)
        for r in results:
            status = "PASS" if r["success"] else "FAIL"
            print(
                f"  [{status}] {r['task_id']:8s}  "
                f"steps={r['steps']:<4d}  score={r['score']:.3f}  "
                f"Q: {r['initial_q']:.3f} -> {r['final_q']:.3f} (d={r['q_delta']:+.3f})",
                file=sys.stderr,
                flush=True,
            )


if __name__ == "__main__":
    asyncio.run(main())
