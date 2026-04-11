"""
Deterministic task graders for layoutenv benchmark tasks.
"""

from dataclasses import dataclass


TASK_SUCCESS_Q_DELTA = {
    "easy": 0.05,
    "medium": 0.1,
    "hard": 0.15,
}


@dataclass(frozen=True)
class TaskGrade:
    task_id: str
    score: float
    success: bool
    q_delta: float


OPEN_INTERVAL_EPS = 5e-2


def _clamp01(x: float) -> float:
    return min(max(x, 0.0), 1.0)


def _clamp_open01(x: float) -> float:
    return min(max(x, OPEN_INTERVAL_EPS), 1.0 - OPEN_INTERVAL_EPS)


def score_from_q_delta(q_delta: float) -> float:
    """
    Map quality delta to a score strictly inside (0, 1).

    OpenEnv submission validation rejects exact 0.0 and 1.0 task
    scores, so we keep the same linear map but clamp into an open
    interval to avoid boundary values under extreme deltas.
    """
    return _clamp_open01(_clamp01((q_delta + 2.0) / 4.0))


def success_from_q_delta(task_id: str, q_delta: float, default_threshold: float) -> bool:
    """
    Determine success using task-specific threshold if available.
    """
    threshold = TASK_SUCCESS_Q_DELTA.get(task_id, default_threshold)
    return q_delta >= threshold


def grade_episode(
    task_id: str,
    initial_quality: float,
    final_quality: float,
    success_q_delta: float = 0.1,
) -> TaskGrade:
    q_delta = final_quality - initial_quality
    return TaskGrade(
        task_id=task_id,
        score=score_from_q_delta(q_delta),
        success=success_from_q_delta(task_id, q_delta, success_q_delta),
        q_delta=q_delta,
    )
