"""
Layout quality metrics for the RL reward signal.

Every metric operates on a list of element dicts in normalised [0,1] space:
    {"cx": float, "cy": float, "w": float, "h": float, "type": str, "font_size": float}

Penalties (lower is better, target 0):    overlap, boundary.
Rewards (higher is better):               alignment, spacing, plausibility.
"""
from itertools import combinations
from typing import Dict, List, Optional
import numpy as np


# Helpers
def _to_ltrb(e: Dict) -> tuple[float, float, float, float]:
    """cxywh to ltrb"""
    hw, hh = e["w"] / 2, e["h"] / 2
    return (e["cx"] - hw, e["cy"] - hh, e["cx"] + hw, e["cy"] + hh)


def _area(e: Dict) -> float:
    return max(e["w"], 0) * max(e["h"], 0)


def _axis_value(e: Dict, axis: str) -> float:
    l, t, r, b = _to_ltrb(e)
    return {"left": l, "right": r, "cx": e["cx"],
            "top": t, "bottom": b, "cy": e["cy"]}[axis]



# Individual metrics
def overlap_score(elements: List[Dict]) -> float:
    """Sum of pairwise intersection / min-area.  0 = no overlap."""
    if len(elements) < 2:
        return 0.0
    total = 0.0
    for a, b in combinations(elements, 2):
        la, ta, ra, ba_ = _to_ltrb(a)
        lb, tb, rb, bb_ = _to_ltrb(b)
        ix = max(0.0, min(ra, rb) - max(la, lb))
        iy = max(0.0, min(ba_, bb_) - max(ta, tb))
        inter = ix * iy
        if inter > 0:
            min_area = min(_area(a), _area(b))
            total += inter / (min_area + 1e-8)
    n_pairs = len(elements) * (len(elements) - 1) / 2
    return total / n_pairs


def boundary_score(elements: List[Dict]) -> float:
    """Fraction of area outside [0,1]^2 per element, averaged.  0 = all inside."""
    if not elements:
        return 0.0
    total = 0.0
    for e in elements:
        l, t, r, b = _to_ltrb(e)
        full_area = _area(e)
        if full_area <= 0:
            continue
        cl = max(l, 0.0)
        ct = max(t, 0.0)
        cr = min(r, 1.0)
        cb = min(b, 1.0)
        clipped_w = max(cr - cl, 0.0)
        clipped_h = max(cb - ct, 0.0)
        clipped_area = clipped_w * clipped_h
        total += 1.0 - (clipped_area / (full_area + 1e-8))
    return total / len(elements)


def alignment_score(elements: List[Dict], eps: float = 0.02) -> float:
    """Fraction of element-pairs that share an aligned edge/centre.  1 = perfect."""
    if len(elements) < 2:
        return 1.0
    axes = ["left", "right", "cx", "top", "bottom", "cy"]
    aligned = 0
    total_pairs = 0
    for axis in axes:
        values = [_axis_value(e, axis) for e in elements]
        for i, j in combinations(range(len(values)), 2):
            total_pairs += 1
            if abs(values[i] - values[j]) < eps:
                aligned += 1
    return aligned / total_pairs if total_pairs > 0 else 0.0


def spacing_score(elements: List[Dict]) -> float:
    """Consistency of vertical and horizontal gaps.  1 = perfectly uniform."""
    if len(elements) < 2:
        return 1.0

    def _gap_consistency(sorted_elems: List[Dict], vertical: bool) -> float:
        gaps = []
        for i in range(len(sorted_elems) - 1):
            a, b = sorted_elems[i], sorted_elems[i + 1]
            if vertical:
                _, _, _, ba = _to_ltrb(a)
                _, tb, _, _ = _to_ltrb(b)
                gaps.append(tb - ba)
            else:
                _, _, ra, _ = _to_ltrb(a)
                lb, _, _, _ = _to_ltrb(b)
                gaps.append(lb - ra)
        if len(gaps) < 2:
            return 1.0
        arr = np.array(gaps)
        mean = np.mean(arr)
        if abs(mean) < 1e-8:
            return 1.0
        cv = np.std(arr) / (abs(mean) + 1e-8)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    by_cy = sorted(elements, key=lambda e: e["cy"])
    by_cx = sorted(elements, key=lambda e: e["cx"])
    v_score = _gap_consistency(by_cy, vertical=True)
    h_score = _gap_consistency(by_cx, vertical=False)
    return (v_score + h_score) / 2.0


def plausibility_score(
    elements: List[Dict],
    stats: Optional[Dict] = None,
) -> float:
    """Gaussian plausibility per element type.  1 = perfect match to data distribution."""
    if not elements or stats is None:
        return 0.0
    total = 0.0
    counted = 0
    for e in elements:
        etype = e.get("type")
        if etype not in stats:
            continue
        mu = stats[etype]["mu"]
        cov_inv = stats[etype]["cov_inv"]
        x = np.array([e["cx"], e["cy"], e["w"], e["h"], e.get("font_size", 0.0)])
        x = np.clip(x, 0.0, 1.0)
        diff = x - mu
        mahal = float(np.sqrt(np.clip(diff @ cov_inv @ diff, 0.0, None)))
        total += float(np.exp(-0.5 * mahal))
        counted += 1
    return total / counted if counted > 0 else 0.0



# Composite quality function
DEFAULT_WEIGHTS = {
    "overlap": 2.0,
    "boundary": 3.0,
    "alignment": 1.0,
    "spacing": 0.5,
    "plausibility": 1.0,
}


def compute_all_metrics(
    elements: List[Dict],
    stats: Optional[Dict] = None,
) -> Dict[str, float]:
    """Return a dict of all individual metric scores."""
    return {
        "overlap": round(overlap_score(elements), 4),
        "boundary": round(boundary_score(elements), 4),
        "alignment": round(alignment_score(elements), 4),
        "spacing": round(spacing_score(elements), 4),
        "plausibility": round(plausibility_score(elements, stats), 4),
    }


def quality_score(
    metrics: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Composite Q(state).  Higher is better."""
    w = weights or DEFAULT_WEIGHTS
    q = (
        -w["overlap"] * metrics["overlap"]
        - w["boundary"] * metrics["boundary"]
        + w["alignment"] * metrics["alignment"]
        + w["spacing"] * metrics["spacing"]
        + w["plausibility"] * metrics["plausibility"]
    )
    return round(q, 4)
