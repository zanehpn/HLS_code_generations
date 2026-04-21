"""
Preference data generation: Select Max & Min → (winner, loser) pairs for DPO.

From the PPA-RTL flow:
- Given multiple designs with scores, select best (max) and worst (min)
- Form preference pairs: winner > loser
- Can also form multiple pairs (e.g., best vs each other, or top-k vs bottom-k)
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import PPAObjective, PPAWeights
from .ppa_evaluator import PPAMetrics, PPAEvaluator
from .scoring import compute_scores_batch


@dataclass
class PreferencePair:
    """A single preference: chosen (winner) > rejected (loser)."""
    prompt: str              # Functional description / instruction
    chosen: str              # HLS design with higher score
    rejected: str            # HLS design with lower score
    chosen_score: float = 0.0
    rejected_score: float = 0.0


def select_max_min_indices(scores: List[float]) -> Tuple[int, int]:
    """Return (argmax, argmin) indices."""
    if not scores:
        raise ValueError("scores cannot be empty")
    max_idx = max(range(len(scores)), key=lambda i: scores[i])
    min_idx = min(range(len(scores)), key=lambda i: scores[i])
    return max_idx, min_idx


def build_preference_pairs(
    prompts: List[str],
    design_groups: List[List[str]],
    objective: PPAObjective = PPAObjective.PPA,
    evaluator: Optional[PPAEvaluator] = None,
) -> List[PreferencePair]:
    """
    Build preference pairs from groups of designs per prompt.

    Each prompt has multiple design candidates. We score them, select max and min,
    and form (chosen, rejected) pairs for DPO.

    Args:
        prompts: Functional descriptions (one per group)
        design_groups: List of design strings per prompt
        objective: PPA optimization objective
        evaluator: PPA evaluator (default: PPAEvaluator())

    Returns:
        List of PreferencePair for DPO training
    """
    if evaluator is None:
        evaluator = PPAEvaluator()
    weights = PPAWeights.from_objective(objective)
    pairs: List[PreferencePair] = []

    for prompt, designs in zip(prompts, design_groups):
        if len(designs) < 2:
            continue
        metrics_list = [evaluator.evaluate(d) for d in designs]
        scores = compute_scores_batch(metrics_list, weights)
        max_idx, min_idx = select_max_min_indices(scores)
        if max_idx == min_idx:
            continue
        pairs.append(PreferencePair(
            prompt=prompt,
            chosen=designs[max_idx],
            rejected=designs[min_idx],
            chosen_score=scores[max_idx],
            rejected_score=scores[min_idx],
        ))
    return pairs


def build_preference_pairs_from_forgehls(
    designs_by_desc: List[Tuple[str, List[dict]]],
    objective: PPAObjective = PPAObjective.PPA,
) -> List[PreferencePair]:
    """
    Build preference pairs from ForgeHLS design dicts.

    designs_by_desc: List of (functional_description, [design_dict, ...])
    """
    evaluator = PPAEvaluator()
    weights = PPAWeights.from_objective(objective)
    pairs: List[PreferencePair] = []

    for desc, designs in designs_by_desc:
        if len(designs) < 2:
            continue
        metrics_list = [evaluator.from_forgehls_json(d) for d in designs]
        scores = compute_scores_batch(metrics_list, weights)
        max_idx, min_idx = select_max_min_indices(scores)
        if max_idx == min_idx:
            continue
        # Extract source code from design
        def get_source(d: dict) -> str:
            src = d.get("source_code", [])
            if isinstance(src, list):
                return "\n".join(
                    f.get("file_content", "") for f in src if isinstance(f, dict)
                )
            return str(src)

        chosen_code = get_source(designs[max_idx])
        rejected_code = get_source(designs[min_idx])
        if not chosen_code or not rejected_code:
            continue
        pairs.append(PreferencePair(
            prompt=desc,
            chosen=chosen_code,
            rejected=rejected_code,
            chosen_score=scores[max_idx],
            rejected_score=scores[min_idx],
        ))
    return pairs


def preference_pairs_to_jsonl(pairs: List[PreferencePair], out_path: str) -> None:
    """Save preference pairs to JSONL for DPO training."""
    import json
    with open(out_path, "w") as f:
        for p in pairs:
            obj = {
                "prompt": p.prompt,
                "chosen": p.chosen,
                "rejected": p.rejected,
                "chosen_score": p.chosen_score,
                "rejected_score": p.rejected_score,
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
