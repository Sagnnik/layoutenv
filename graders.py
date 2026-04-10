"""
Hackathon-facing grader adapters.

These wrappers expose one callable per task in a conventional top-level module.
They delegate to the canonical grading logic in grader.py while accepting a
variety of payload shapes that hidden validators may provide.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from grader import TaskGrade, grade_episode, score_from_q_delta


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


def _build_grade_dict(task_id: str, q_delta: Optional[float]) -> Dict[str, Any]:
    safe_q_delta = 0.0 if q_delta is None else q_delta
    if q_delta is None:
        score = score_from_q_delta(safe_q_delta)
        passed = False
        return {
            "task_id": task_id,
            "score": score,
            "passed": passed,
            "success": passed,
            "q_delta": safe_q_delta,
            "breakdown": {"q_delta": safe_q_delta},
        }

    grade: TaskGrade = grade_episode(
        task_id=task_id,
        initial_quality=0.0,
        final_quality=safe_q_delta,
    )
    return {
        "task_id": task_id,
        "score": grade.score,
        "passed": grade.success,
        "success": grade.success,
        "q_delta": grade.q_delta,
        "breakdown": {"q_delta": grade.q_delta},
    }


def _resolve_q_delta(result: Any, kwargs: Dict[str, Any]) -> Optional[float]:
    q_delta = _extract_q_delta(kwargs)
    if q_delta is not None:
        return q_delta
    return _extract_q_delta(result)


def grade_easy(result: Any = None, **kwargs: Any) -> Dict[str, Any]:
    return _build_grade_dict("easy", _resolve_q_delta(result, kwargs))


def grade_medium(result: Any = None, **kwargs: Any) -> Dict[str, Any]:
    return _build_grade_dict("medium", _resolve_q_delta(result, kwargs))


def grade_hard(result: Any = None, **kwargs: Any) -> Dict[str, Any]:
    return _build_grade_dict("hard", _resolve_q_delta(result, kwargs))
