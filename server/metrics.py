"""
Layout quality metrics for the RL reward signal.

Every metric operates on a list of element dicts in normalised [0,1] space:
    {"cx": float, "cy": float, "w": float, "h": float, "type": str, "font_size": float}

Penalties (lower is better, target 0):    overlap, boundary, occlusion.
Rewards (higher is better):               alignment, spacing, plausibility.

Time Complexities (maybe I should fix them later):
- overlap_score: O(n^2)
- boundary_score: O(n)
- alignment_score: O(n^2)
- spacing_score: O(nlogn)
- plausibility_score: O(n)
- occlusion_score: O(n*h*w)
- compute_all_metrics: O(n)
- quality_score: O(1)
"""
from itertools import combinations
from typing import Dict, List, Optional, Set
import numpy as np


# =============================================================================
# Helpers
# =============================================================================
def _precompute_boxes(elements: List[Dict]) -> np.ndarray:
    """Convert list of element dicts to (n, 4) float32 array of [l, t, r, b]."""
    n = len(elements)
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i, e in enumerate(elements):
        hw = e["w"] * 0.5
        hh = e["h"] * 0.5
        boxes[i] = [e["cx"] - hw, e["cy"] - hh, e["cx"] + hw, e["cy"] + hh]
    return boxes


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


# =============================================================================
# Individual Metrics
# =============================================================================
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

    boxes = _precompute_boxes(elements)
    areas = np.array([e["w"] * e["h"] for e in elements], dtype=np.float32)

    l = np.clip(boxes[:, 0], 0.0, 1.0)
    t = np.clip(boxes[:, 1], 0.0, 1.0)
    r = np.clip(boxes[:, 2], 0.0, 1.0)
    b = np.clip(boxes[:, 3], 0.0, 1.0)

    clipped_areas = np.maximum(r - l, 0.0) * np.maximum(b - t, 0.0)

    valid = areas > 0
    ratios = np.zeros_like(areas)
    ratios[valid] = 1.0 - (clipped_areas[valid] / areas[valid])

    return float(np.mean(ratios))


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

    boxes = _precompute_boxes(elements)

    cy = np.array([e["cy"] for e in elements])
    sort_idx = np.argsort(cy)
    sorted_boxes = boxes[sort_idx]
    v_gaps = sorted_boxes[1:, 1] - sorted_boxes[:-1, 3]  # top[i+1] - bottom[i]

    cx = np.array([e["cx"] for e in elements])
    sort_idx = np.argsort(cx)
    sorted_boxes = boxes[sort_idx]
    h_gaps = sorted_boxes[1:, 0] - sorted_boxes[:-1, 2]  # left[i+1] - right[i]

    def _consistency(gaps: np.ndarray) -> float:
        if len(gaps) < 2:
            return 1.0
        mean = float(np.mean(gaps))
        if abs(mean) < 1e-8:
            return 1.0
        cv = float(np.std(gaps)) / (abs(mean) + 1e-8)
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    return (_consistency(v_gaps) + _consistency(h_gaps)) * 0.5


def plausibility_score(
    elements: List[Dict],
    stats: Optional[Dict] = None,
) -> float:
    """Batched Gaussian plausibility per element type.  1 = perfect match to data distribution."""
    if not elements or stats is None:
        return 0.0

    features = np.zeros((len(elements), 5), dtype=np.float32)
    for i, e in enumerate(elements):
        features[i] = [e["cx"], e["cy"], e["w"], e["h"], e.get("font_size", 0.0)]
    features = np.clip(features, 0.0, 1.0)

    type_groups: Dict[str, List[int]] = {}
    for i, e in enumerate(elements):
        etype = e.get("type")
        if etype in stats:
            type_groups.setdefault(etype, []).append(i)

    total_score = 0.0
    total_counted = 0

    for etype, indices in type_groups.items():
        mu = stats[etype]["mu"]
        cov_inv = stats[etype]["cov_inv"]
        diff = features[indices] - mu        # (k, 5)
        left = diff @ cov_inv                # (k, 5)
        mahal = np.sqrt(np.clip(np.einsum('ij,ij->i', left, diff), 0.0, None))
        total_score += float(np.sum(np.exp(-0.5 * mahal)))
        total_counted += len(indices)

    return total_score / total_counted if total_counted > 0 else 0.0


def occlusion_score(
    elements: List[Dict],
    saliency_map: Optional[np.ndarray],
) -> Optional[float]:
    """
    Saliency-covered ratio by layout elements.  0 = no salient area covered.

    Returns None when saliency is unavailable/invalid so callers can apply a
    neutral fallback policy without biasing the reward.
    """
    if saliency_map is None:
        return None

    sal = np.asarray(saliency_map, dtype=np.float32)
    if sal.ndim != 2 or sal.size == 0:
        return None

    sal = np.nan_to_num(sal, nan=0.0, posinf=0.0, neginf=0.0)
    sal = np.clip(sal, 0.0, None)
    total_saliency = float(np.sum(sal))
    if total_saliency <= 1e-8:
        return 0.0

    h, w = sal.shape
    mask = np.zeros((h, w), dtype=bool)

    boxes = _precompute_boxes(elements)
    for l, t, r, b in boxes:
        l = max(0.0, min(1.0, float(l)))
        t = max(0.0, min(1.0, float(t)))
        r = max(0.0, min(1.0, float(r)))
        b = max(0.0, min(1.0, float(b)))
        if r <= l or b <= t:
            continue
        x1 = max(0, min(w, int(np.floor(l * w))))
        x2 = max(0, min(w, int(np.ceil(r * w))))
        y1 = max(0, min(h, int(np.floor(t * h))))
        y2 = max(0, min(h, int(np.ceil(b * h))))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True

    if not np.any(mask):
        return 0.0

    return float(np.clip(np.sum(sal[mask]) / total_saliency, 0.0, 1.0))


# =============================================================================
# Composite Quality Function
# =============================================================================
DEFAULT_WEIGHTS = {
    "overlap": 2.0,
    "boundary": 3.0,
    "occlusion": 1.0,
    "alignment": 1.0,
    "spacing": 0.5,
    "plausibility": 1.0,
}


def compute_all_metrics(
    elements: List[Dict],
    stats: Optional[Dict] = None,
    saliency_map: Optional[np.ndarray] = None,
    content_metric_names: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Return a dict of all individual metric scores."""
    content_metric_names = content_metric_names or set()
    occ: Optional[float]
    if "occlusion" in content_metric_names:
        occ = occlusion_score(elements, saliency_map)
    else:
        occ = None

    occ_out = 0.0 if occ is None else occ
    return {
        "overlap": round(overlap_score(elements), 4),
        "boundary": round(boundary_score(elements), 4),
        "occlusion": round(occ_out, 4),
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
        - w["occlusion"] * metrics["occlusion"]
        + w["alignment"] * metrics["alignment"]
        + w["spacing"] * metrics["spacing"]
        + w["plausibility"] * metrics["plausibility"]
    )
    return round(q, 4)
