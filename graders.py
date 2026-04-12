"""
Hackathon-facing grader adapters.

This module has ZERO external dependencies — no openenv, no FastAPI, no
project-local modules. This makes it instantly importable by the hidden
validator without triggering import errors or framework bootstrapping.

Each grader function returns a plain float strictly inside (0, 1).
"""

from __future__ import annotations

from typing import Any, Optional


# ── scoring logic (self-contained, no imports from grader.py) ────────────────

_EPS = 0.01

_TASK_SUCCESS_Q_DELTA = {
    "easy": 0.10,
    "medium": 0.20,
    "hard": 0.32,
}


def _safe_score(raw: float) -> float:
    """Clamp a raw score strictly inside (0, 1) and round to 2 dp."""
    return round(min(max(float(raw), _EPS), 1.0 - _EPS), 2)


def _score_from_q_delta(q_delta: float) -> float:
    """Map quality delta to a score strictly inside (0, 1)."""
    linear = (q_delta + 2.0) / 4.0
    clamped = min(max(linear, 0.0), 1.0)          # [0, 1]
    return min(max(clamped, 0.05), 0.95)           # (0, 1) with margin


# ── payload extraction helpers ───────────────────────────────────────────────

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
        return _safe_score(_score_from_q_delta(0.0))
    return _safe_score(_score_from_q_delta(q_delta))


def grade_medium(result: Any = None, **kwargs: Any) -> float:
    q_delta = _resolve_q_delta(result, kwargs)
    if q_delta is None:
        return _safe_score(_score_from_q_delta(0.0))
    return _safe_score(_score_from_q_delta(q_delta))


def grade_hard(result: Any = None, **kwargs: Any) -> float:
    q_delta = _resolve_q_delta(result, kwargs)
    if q_delta is None:
        return _safe_score(_score_from_q_delta(0.0))
    return _safe_score(_score_from_q_delta(q_delta))
