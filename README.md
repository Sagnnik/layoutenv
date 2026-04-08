---
title: LayoutEnv Environment Server
emoji: 🎭
colorFrom: green
colorTo: yellow
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# LayoutEnv: Poster Layout Refinement Environment

`layoutenv` is a practical OpenEnv benchmark for iterative poster/layout cleanup.
An agent receives a noisy layout and improves it step-by-step using discrete edit actions:
`MOVE`, `RESIZE`, `ALIGN`, `SNAP`, and `NO_OP`.

The task is designed for:
- spatial reasoning across multiple elements
- optimization with shaped rewards
- LLM and VLM agent evaluation on iterative improvement loops

## Task Overview

Each episode starts from a perturbed sample with normalized geometry.
At every step, the agent picks:
- target element (`element_id`)
- action type (`MOVE`, `RESIZE`, `ALIGN`, `SNAP`, `NO_OP`)
- action parameter (`UP`, `LEFT`, `CENTER_X`, `GRID`, etc.)
- optional magnitude for `MOVE`/`RESIZE` (`SMALL`, `MEDIUM`, `LARGE`)

The episode ends when:
- max step budget is reached, or
- agent emits `NO_OP` (treat as stop)

## Quick Start

The simplest way to use the environment is through the `LayoutEnv` client:

```python
from layoutenv import LayoutAction, LayoutEnv

async def run_example() -> None:
    env = await LayoutEnv.from_docker_image("layoutenv:latest")
    try:
        result = await env.reset(mode="llm")
        print("Initial Q:", result.observation.quality_score)
        result = await env.step(LayoutAction(
            element_id=0,
            action="ALIGN",
            param="CENTER_X",
            magnitude="MEDIUM",
        ))
        print("Reward:", result.reward, "Done:", result.done)
    finally:
        await env.close()

import asyncio
asyncio.run(run_example())
```

If you prefer a sync client flow, instantiate with `LayoutEnv(base_url=...)`
and call the synchronous methods in your own wrapper.

`LayoutEnv.from_docker_image(...)` handles:
- starting the container
- waiting for readiness
- connecting the client
- container cleanup on `close()`

## Build the Docker Image

From repo root:

```bash
docker build -t layoutenv:latest -f layoutenv/server/Dockerfile .
```

From `layoutenv/` directory:

```bash
docker build -t layoutenv:latest -f server/Dockerfile .
```

## Run the Server (Volume-Mounted Dataset)

The current runtime expects dataset assets at `/app/env/dataset` in-container.
Recommended run command from repo root:

```bash
docker run --rm -d \
  --name layoutenv-server \
  -p 8000:8000 \
  -v "$(pwd)/dataset:/app/env/dataset" \
  layoutenv:latest
```

Verify endpoints:

```bash
curl -s http://localhost:8000/health
curl -s -X POST -H "Content-Type: application/json" -d '{}' http://localhost:8000/reset
```

Stop:

```bash
docker stop layoutenv-server
```

## Usage

Submission baseline script is root `inference.py` (required location).
It uses `LayoutEnv.from_docker_image()` and emits evaluator-friendly stdout:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

### LLM run

```bash
API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... \
IMAGE_NAME=layoutenv:latest python inference.py --seed 42
```

### VLM easy-task smoke

```bash
API_BASE_URL=... MODEL_NAME=... HF_TOKEN=... \
IMAGE_NAME=layoutenv:latest python inference.py --mode vlm --task easy --max-steps 5 --seed 42
```

## Environment Details

### Action (`LayoutAction`)

Fields:
- `element_id` (int): target element index
- `action` (str): `MOVE` | `RESIZE` | `ALIGN` | `SNAP` | `NO_OP`
- `param` (str):
  - `MOVE`: `UP`, `DOWN`, `LEFT`, `RIGHT`
  - `RESIZE`: `WIDER`, `NARROWER`, `TALLER`, `SHORTER`
  - `ALIGN`: `LEFT`, `CENTER_X`, `RIGHT`, `TOP`, `CENTER_Y`, `BOTTOM`
  - `SNAP`: `GRID`
  - `NO_OP`: `NONE`
- `magnitude` (str): `SMALL`, `MEDIUM`, `LARGE` (used for `MOVE`/`RESIZE`)

### Observation (`LayoutObservation`)

Per-step payload includes:
- `canvas`: normalized canvas (`width=1.0`, `height=1.0`)
- `elements`: list of `{id, type, cx, cy, w, h, font_size}`
- `metrics`: layout metrics:
  - `overlap` (lower better)
  - `boundary` (lower better)
  - `alignment` (higher better)
  - `spacing` (higher better)
  - `plausibility` (higher better)
- `quality_score`: composite quality value `Q`
- `initial_quality_score`: `Q` at reset
- `step`, `max_steps`
- optional VLM fields (`image_path`, `rendered_image_base64`)
- optional `text_feedback`

### State (`LayoutState`)

Server state tracks:
- `episode_id`, `step_count`
- current `elements`
- `previous_quality`, `initial_quality`
- VLM context (`current_image_rel`, `dataset_json_path`)

### Reward

Step reward is shaped by quality improvements:
- `reward = REWARD_SCALE * (Q_t - Q_{t-1}) + STEP_PENALTY`
- invalid actions incur a penalty
- terminal shaping applies at episode end

This gives dense training/evaluation signal, not only terminal success.

## Task Grading

Deterministic grading logic is implemented in `layoutenv/grader.py`:
- `q_delta = final_q - initial_q`
- `score = clamp((q_delta + 2.0) / 4.0, 0, 1)`
- task-specific success thresholds:
  - `easy >= 0.05`
  - `medium >= 0.10`
  - `hard >= 0.15`

Note: score is intentionally clamped to `[0, 1]` for stable reporting.

## Deploy to Hugging Face Spaces

From `layoutenv/`:

```bash
openenv push
```

Or specify repo:

```bash
openenv push --repo-id <namespace>/<space-name>
```

After deploy, verify:
- `POST /reset` returns 200
- `/docs` is reachable
- `/health` is healthy

## Project Structure

```text
layoutenv/
├── __init__.py
├── client.py
├── grader.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
└── server/
    ├── app.py
    ├── layout_environment.py
    ├── metrics.py
    └── Dockerfile
```
