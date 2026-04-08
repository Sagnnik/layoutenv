"""
Find representative task samples at three difficulty levels from the dataset.

Easy   : 3–4  elements, noise=0.05
Medium : 5–7  elements, noise=0.10
Hard   : 7–10 elements, noise=0.15

Writes dataset/task_samples.json for inference.py, copies referenced PNGs into
dataset/sample_images/, and rewrites image_path / layer_image_path in the JSON
to point at sample_images/ (paths relative to the dataset directory).
"""

from __future__ import annotations

import copy
import json
import random
import shutil
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_JSON = SCRIPT_DIR / "genposter_5000_images.json"
OUTPUT_JSON = SCRIPT_DIR / "task_samples.json"
SAMPLE_IMAGES_DIR = SCRIPT_DIR / "sample_images"
SAMPLE_IMAGES_PREFIX = "sample_images"

TASKS = [
    {
        "task_id": "easy",
        "min_elements": 3,
        "max_elements": 4,
        "noise": 0.05,
        "max_steps": 50,
    },
    {
        "task_id": "medium",
        "min_elements": 5,
        "max_elements": 7,
        "noise": 0.10,
        "max_steps": 100,
    },
    {
        "task_id": "hard",
        "min_elements": 7,
        "max_elements": 10,
        "noise": 0.15,
        "max_steps": 200,
    },
]


def _copy_and_remap_path(
    rel: str,
    sample_images_dir: Path,
    dataset_root: Path,
    missing: list[str],
) -> str:
    name = Path(rel).name
    src = (dataset_root / rel).resolve()
    if not src.is_file():
        missing.append(rel)
        return f"{SAMPLE_IMAGES_PREFIX}/{name}"

    dest = sample_images_dir / name
    shutil.copy2(src, dest)
    return f"{SAMPLE_IMAGES_PREFIX}/{name}"


def copy_media_and_rewrite_paths(
    output: list[dict],
    *,
    sample_images_dir: Path = SAMPLE_IMAGES_DIR,
    dataset_root: Path = SCRIPT_DIR,
) -> None:
    if sample_images_dir.exists():
        shutil.rmtree(sample_images_dir)
    sample_images_dir.mkdir(parents=True)

    missing: list[str] = []
    for entry in output:
        sample = entry["sample"]
        ip = sample.get("image_path")
        if ip:
            sample["image_path"] = _copy_and_remap_path(
                ip, sample_images_dir, dataset_root, missing
            )
        for el in sample.get("elements") or []:
            lp = el.get("layer_image_path")
            if lp:
                el["layer_image_path"] = _copy_and_remap_path(
                    lp, sample_images_dir, dataset_root, missing
                )

    if missing:
        for m in missing:
            print(f"WARNING: missing source file: {dataset_root / m}", file=sys.stderr)


def main() -> None:
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    rng = random.Random(42)

    output: list[dict] = []
    for task in TASKS:
        candidates = [
            s
            for s in dataset
            if task["min_elements"] <= len(s.get("elements", [])) <= task["max_elements"]
            and s.get("image_path")
        ]
        if not candidates:
            candidates = [
                s
                for s in dataset
                if task["min_elements"] <= len(s.get("elements", [])) <= task["max_elements"]
            ]
        if not candidates:
            print(
                f"WARNING: No samples found for task '{task['task_id']}' "
                f"({task['min_elements']}–{task['max_elements']} elements). Skipping."
            )
            continue

        chosen = rng.choice(candidates)
        output.append(
            {
                "task_id": task["task_id"],
                "noise": task["noise"],
                "max_steps": task["max_steps"],
                "sample": copy.deepcopy(chosen),
            }
        )
        print(
            f"  {task['task_id']:8s}  id={chosen['id']:<6}  "
            f"elements={len(chosen['elements'])}  noise={task['noise']}"
        )

    copy_media_and_rewrite_paths(output)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nCopied media to {SAMPLE_IMAGES_DIR}")
    print(f"Saved {len(output)} task samples to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
