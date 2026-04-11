"""
Hackathon-facing grader adapters.

Each grader function accepts arbitrary positional/keyword arguments
(the hidden validator may pass different payload shapes) and returns
a **plain float** strictly inside (0, 1).

OpenEnv submission validation rejects exact 0.0 and 1.0 task scores.
"""

from __future__ import annotations

from typing import Any, Optional

from grader import grade_episode, score_from_q_delta


# ── helpers ──────────────────────────────────────────────────────────────────

_EPS = 0.01


def _safe_score(raw: float) -> float:
    """Clamp a raw score strictly inside (0, 1) and round to 2 dp."""
    return round(min(max(float(raw), _EPS), 1.0 - _EPS), 2)


def _maybe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_q_delta(payload: Any) -> Optional[float]:
    if payload is None:
        return None

    if isinstance(payload, (int, float)):
        return float(payload)

    if isinstance(payload, dict):
        for key in ("q_delta", "delta_q", "quality_delta"):
            value = _maybe_float(payload.get(key))
            if value is not None:
                return value

        initial_q = _maybe_float(
            payload.get("initial_quality")
            or payload.get("initial_q")
            or payload.get("initial_quality_score")
        )
        final_q = _maybe_float(
            payload.get("final_quality")
            or payload.get("final_q")
            or payload.get("final_quality_score")
            or payload.get("quality_score")
        )
        if initial_q is not None and final_q is not None:
            return final_q - initial_q

        observation = payload.get("observation")
        if isinstance(observation, dict):
            obs_delta = _extract_q_delta(observation)
            if obs_delta is not None:
                return obs_delta

    return None


def _resolve_q_delta(result: Any, kwargs: dict) -> Optional[float]:
    q_delta = _extract_q_delta(kwargs)
    if q_delta is not None:
        return q_delta
    return _extract_q_delta(result)


# ── public grader functions (return float) ───────────────────────────────────


def grade_easy(result: Any = None, **kwargs: Any) -> float:
    q_delta = _resolve_q_delta(result, kwargs)
    if q_delta is None:
        return _safe_score(score_from_q_delta(0.0))
    grade = grade_episode(task_id="easy", initial_quality=0.0, final_quality=q_delta)
    return _safe_score(grade.score)


def grade_medium(result: Any = None, **kwargs: Any) -> float:
    q_delta = _resolve_q_delta(result, kwargs)
    if q_delta is None:
        return _safe_score(score_from_q_delta(0.0))
    grade = grade_episode(task_id="medium", initial_quality=0.0, final_quality=q_delta)
    return _safe_score(grade.score)


def grade_hard(result: Any = None, **kwargs: Any) -> float:
    q_delta = _resolve_q_delta(result, kwargs)
    if q_delta is None:
        return _safe_score(score_from_q_delta(0.0))
    grade = grade_episode(task_id="hard", initial_quality=0.0, final_quality=q_delta)
    return _safe_score(grade.score)
