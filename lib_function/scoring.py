"""
PPA scoring with configurable weights.

Score = w_P * (1 - norm_power) + w_P' * (1 - norm_perf) + w_A * (1 - norm_area)
Lower PPA is better, so we invert for "higher score = better".
"""

from typing import List, Optional, Tuple

from .config import PPAObjective, PPAWeights
from .ppa_evaluator import PPAMetrics


def normalize_metric(values: List[float], v: float, lower_better: bool = True) -> float:
    """
    Min-max normalize a single value given a list of values.
    lower_better: True for power, perf, area (lower is better).
    Returns value in [0, 1] where 1 means best (lowest when lower_better).
    """
    if not values or v != v:  # NaN check
        return 0.5
    lo, hi = min(values), max(values)
    if hi == lo:
        return 1.0
    norm = (v - lo) / (hi - lo)
    if lower_better:
        return 1.0 - norm  # best (lowest) -> 1
    return norm  # best (highest) -> 1


def compute_score(
    metrics: PPAMetrics,
    weights: PPAWeights,
    all_power: Optional[List[float]] = None,
    all_perf: Optional[List[float]] = None,
    all_area: Optional[List[float]] = None,
) -> float:
    """
    Compute weighted PPA score. Higher score = better design.

    If normalization lists are not provided, uses raw inverse (1/value) with clamping.
    """
    if weights.w_power + weights.w_perf + weights.w_area == 0:
        return 0.0

    def safe_norm(vals: Optional[List[float]], v: float) -> float:
        if vals is not None:
            return normalize_metric(vals, v, lower_better=True)
        # Fallback: use 1/(1+x) to map positive values to (0,1]
        if v <= 0:
            return 0.0
        return 1.0 / (1.0 + v / 1000.0)

    s_power = safe_norm(all_power, metrics.power_mw)
    s_perf = safe_norm(all_perf, metrics.perf_ns)
    s_area = safe_norm(all_area, metrics.area_um2)

    total_w = weights.w_power + weights.w_perf + weights.w_area
    score = (
        weights.w_power * s_power
        + weights.w_perf * s_perf
        + weights.w_area * s_area
    ) / total_w
    return max(0.0, min(1.0, score))


def compute_scores_batch(
    metrics_list: List[PPAMetrics],
    weights: PPAWeights,
) -> List[float]:
    """Compute scores for a batch of designs with shared normalization."""
    powers = [m.power_mw for m in metrics_list]
    perfs = [m.perf_ns for m in metrics_list]
    areas = [m.area_um2 for m in metrics_list]
    return [
        compute_score(m, weights, powers, perfs, areas)
        for m in metrics_list
    ]
