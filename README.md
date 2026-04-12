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

## TL;DR

`LayoutEnv` is an RL environment for improving noisy UI/poster layouts via iterative edit actions.

* **Task**: Refine a perturbed layout into a clean, structured design
* **State**: Structured layout (positions, sizes, element types)
* **Actions**: MOVE, RESIZE, ALIGN, SNAP, NO_OP
* **Reward**: Change in layout quality score (alignment, spacing, overlap, occlusion)

✔ Dense reward signal
✔ Multi-step structured decision making
✔ Content-aware (occlusion-based) evaluation for VLM agents

---

`layoutenv` is a practical OpenEnv benchmark for iterative poster/layout cleanup.
An agent receives a noisy layout and improves it step-by-step using discrete edit actions.

The task is designed for:

* spatial reasoning across multiple elements
* optimization with shaped rewards
* LLM and VLM agent evaluation on iterative improvement loops
* **content-aware layout optimization via occlusion scoring**

---

---

## Why this is a Good RL Environment

- **Interdependent State Space**: Moving or resizing one element often affects the metrics of others (e.g., fixing an overlap might break alignment or spacing consistency), forcing the agent to learn holistic layout strategies rather than greedy local optimizations.
- **Mixed Action Complexity**: The environment combines low-level positioning (`MOVE`, `RESIZE`) with high-level structural primitives (`ALIGN`, `SNAP`), allowing for research into hierarchical policy learning and action selection.
- **Visual & Semantic Modalities**: Supports both coordinate-based (LLM) and raw image/rendering-based (VLM) observation spaces, making it a versatile benchmark for multi-modal agents.
- **Real-World Utility**: Layout refinement is a high-value task in design automation, presenting a clear path from RL training to practical application in creative tools.

---

## Learning Signal (Proof of Concept)

The environment provides a meaningful optimization signal:
- **Random actions** → unstable or low quality score
- **Structured actions** (e.g., ALIGN, SNAP) → consistent improvement in Q
- **Reward directly correlates** with layout quality improvements

This demonstrates that agents can learn policies that improve layout structure over time.

---

## Optimization Objective

The primary goal is to maximize the **Composite Quality Score ($Q$)**, which evaluates several core design principles:

### Penalties (Lower is better)
- **Overlap**: Pairwise element intersections.
- **Boundary**: Keeping all elements within the [0, 1] canvas.
- **Occlusion**: Covering visually important regions (content-aware).

### Rewards (Higher is better)
- **Alignment**: Shared edges and centers between elements.
- **Spacing**: Consistency of horizontal and vertical gaps.
- **Plausibility**: Statistical adherence to standard design distributions.

The agent receives a shaped reward $R_t = 10 \cdot (Q_t - Q_{t-1}) - 0.05$, incentivizing steady quality improvements while discouraging inefficient step usage. A terminal bonus is awarded for achieving significant overall improvement ($Q_{final} - Q_{initial} \ge 0.05$).

## Task Overview

Each episode starts from a perturbed layout.

At every step, the agent selects:

* target element (`element_id`)
* action type (`MOVE`, `RESIZE`, `ALIGN`, `SNAP`, `NO_OP`)
* action parameter (`UP`, `LEFT`, `CENTER_X`, etc.)
* optional magnitude (`SMALL`, `MEDIUM`, `LARGE`)

Episode ends when:

* step limit reached, or
* agent emits `NO_OP`

---

## RL Interaction Loop

1. Observe current layout
2. Select an edit action
3. Environment updates layout
4. Receive reward based on quality improvement

The agent must learn a sequence of edits that maximizes cumulative reward.

---

## Quick Start

```python
from layoutenv import LayoutAction, LayoutEnv

async def run_example():
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
```

---

## Build the Docker Image

```bash
docker build -t layoutenv:latest -f Dockerfile .
```

---

## Run the Server

```bash
docker run --rm -d \
  --name layoutenv-server \
  -p 8000:8000 \
  -v "$(pwd)/dataset:/app/env/dataset" \
  layoutenv:latest
```

---

## Environment Details

### Observation (`LayoutObservation`)

Includes:

* element geometry and types
* layout metrics:

  * overlap (↓)
  * boundary (↓)
  * **occlusion (↓)**
  * alignment (↑)
  * spacing (↑)
  * plausibility (↑)
* composite quality score `Q`

---

## Saliency-Aware Occlusion (VLM Mode)

In `mode="vlm"`, the environment supports content-aware evaluation.

* Uses saliency maps (`.npy`)
* Penalizes covering important visual regions
* Encourages agents to avoid placing elements over:

  * faces
  * focal objects
  * key content areas

This introduces **semantic reasoning into layout optimization**.

---

## Reward

```
reward = REWARD_SCALE * (Q_t - Q_{t-1}) + STEP_PENALTY
```

* Dense signal (not sparse)
* Penalizes invalid actions
* Encourages consistent improvement

---

## Task Grading

* `q_delta = final_q - initial_q`
* normalized score with margin clamping

Thresholds:

* easy ≥ 0.10
* medium ≥ 0.20
* hard ≥ 0.32

---

## Project Structure

```text
.
├── Dockerfile
├── client.py
├── grader.py
├── inference.py
├── models.py
├── server/
│   ├── app.py
│   ├── layout_environment.py
│   └── metrics.py
```
