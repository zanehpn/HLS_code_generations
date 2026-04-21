"""
Dataset loading for SFT and DPO training.

- SFT: (instruction, response) pairs for NL2HLS
- DPO: (prompt, chosen, rejected) preference pairs
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from datasets import Dataset


# Chat template for instruction-following
SFT_SYSTEM = (
    "You are an expert in High-Level Synthesis (HLS). "
    "Given a functional description, generate optimized C/C++ HLS code with appropriate pragmas."
)

DPO_SYSTEM = SFT_SYSTEM


def format_sft_example(instruction: str, response: str, system: str = SFT_SYSTEM) -> str:
    """Format as chat for SFT (e.g., Qwen/ChatML style)."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n{response}<|im_end|>"
    )


def format_sft_prompt(instruction: str, system: str = SFT_SYSTEM) -> str:
    """Prompt prefix up to assistant header (without response)."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _coerce_metric_value(value: object) -> Optional[float]:
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    if metric != metric or metric in (float("inf"), float("-inf")):
        return None
    return metric


def _within_resource_limits(metadata: Dict, resource_limits: Optional[Dict[str, float]]) -> bool:
    if not resource_limits:
        return True
    for metric_name, metric_limit in resource_limits.items():
        metric_value = _coerce_metric_value(metadata.get(metric_name))
        if metric_value is None or metric_value > metric_limit:
            return False
    return True


def load_sft_jsonl(path: Union[str, Path], resource_limits: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Load SFT data from JSONL and optionally drop examples that exceed resource limits."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support multiple formats: instruction/response, prompt/completion, input/output
            inst = obj.get("instruction") or obj.get("prompt") or obj.get("input", "")
            resp = obj.get("response") or obj.get("completion") or obj.get("output") or obj.get("target", "")
            metadata = obj.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}
            if inst and resp and _within_resource_limits(metadata, resource_limits):
                out.append({"instruction": inst, "response": resp})
    return out


def _designs_to_dpo_pair(chosen_design: Dict, rejected_design: Dict, metric: str = "Worst-caseLatency") -> Dict:
    """Create a DPO preference pair from two designs (chosen has lower metric)."""
    input_text = chosen_design.get("input", "")
    metric_to_goal = {
        "Worst-caseLatency": "with less latency",
        "Best-caseLatency": "with less latency",
        "LUT": "with fewer LUT resources",
        "DSP": "with fewer DSP resources",
        "BRAM_18K": "with less BRAM usage",
        "FF": "with fewer flip-flops",
    }
    optimization_goal = metric_to_goal.get(metric, "with better performance")
    if "Generate HLS code" in input_text:
        parts = input_text.split("Generate HLS code")
        functional_desc = parts[0].strip()
        if functional_desc.startswith("Functional Description:"):
            functional_desc = functional_desc.replace("Functional Description:", "").strip()
        algo_name = chosen_design.get("metadata", {}).get("algo_name", "algorithm")
        source_name = chosen_design.get("metadata", {}).get("source_name", "")
        prompt = (
            f"{functional_desc}\n\nGenerate optimized HLS code for the {algo_name} algorithm (from {source_name}) {optimization_goal}."
            if source_name
            else f"{functional_desc}\n\nGenerate optimized HLS code for the {algo_name} algorithm {optimization_goal}."
        )
    else:
        prompt = input_text
    return {
        "prompt": prompt,
        "chosen": chosen_design.get("output", ""),
        "rejected": rejected_design.get("output", ""),
    }


def load_dpo_from_designs_jsonl(
    path: Union[str, Path],
    metric: str = "Worst-caseLatency",
    pairing_strategy: str = "best_vs_worst",
) -> List[Dict]:
    """Load designs JSONL and generate DPO preference pairs (for eval)."""
    from collections import defaultdict

    path = Path(path)
    if not path.exists():
        return []
    designs = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    designs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    groups = defaultdict(list)
    for d in designs:
        meta = d.get("metadata", {})
        key = (meta.get("source_name", "unknown"), meta.get("algo_name", "unknown"))
        groups[key].append(d)
    pairs = []
    for group_designs in groups.values():
        if len(group_designs) < 2:
            continue
        valid = []
        for d in group_designs:
            meta = d.get("metadata", {})
            v = meta.get(metric)
            if v is not None:
                try:
                    v = float(v)
                    if v > 0:
                        valid.append((v, d))
                except (ValueError, TypeError):
                    pass
        if len(valid) < 2:
            continue
        valid.sort(key=lambda x: x[0])
        if pairing_strategy == "best_vs_worst":
            if valid[0][0] < valid[-1][0]:
                pairs.append(_designs_to_dpo_pair(valid[0][1], valid[-1][1], metric))
    return pairs


def load_dpo_jsonl(path: Union[str, Path]) -> List[Dict]:
    """Load DPO preference data from JSONL."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompt = obj.get("prompt", "")
            chosen = obj.get("chosen", "")
            rejected = obj.get("rejected", "")
            if prompt and chosen and rejected:
                out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return out


def sft_data_to_hf_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 2048,
    map_batch_size: int = 256,
) -> Dataset:
    """
    Convert SFT list to HuggingFace Dataset with assistant-only labels.

    Loss is computed only on assistant response tokens.
    System/user/prompt prefix and padding tokens are masked with -100.
    """
    raw_ds = Dataset.from_list(data)

    def _tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        prompts = [format_sft_prompt(inst) for inst in batch["instruction"]]
        full_texts = [p + resp + "<|im_end|>" for p, resp in zip(prompts, batch["response"])]

        full_tok = tokenizer(
            full_texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )
        prompt_tok = tokenizer(
            prompts,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None,
        )

        labels: List[List[int]] = []
        for input_ids, attn_mask, prompt_ids in zip(
            full_tok["input_ids"],
            full_tok["attention_mask"],
            prompt_tok["input_ids"],
        ):
            lab = list(input_ids)
            prompt_len = min(len(prompt_ids), len(lab))
            for i in range(prompt_len):
                lab[i] = -100
            for i, m in enumerate(attn_mask):
                if m == 0:
                    lab[i] = -100
            labels.append(lab)

        full_tok["labels"] = labels
        return full_tok

    return raw_ds.map(
        _tokenize_batch,
        batched=True,
        batch_size=map_batch_size,
        remove_columns=raw_ds.column_names,
        desc="Tokenizing SFT data (assistant-only labels)",
    )


def dpo_data_to_hf_dataset(
    data: List[Dict],
    tokenizer,
    max_length: int = 4096,
    max_prompt_length: int = 1024,
) -> Dataset:
    """Convert DPO preference data to HF Dataset for DPO trainer."""
    prompts = [d["prompt"] for d in data]
    chosen = [d["chosen"] for d in data]
    rejected = [d["rejected"] for d in data]

    def _tokenize_prompt(p: str):
        msg = (
            f"<|im_start|>system\n{DPO_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{p}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return tokenizer(
            msg,
            truncation=True,
            max_length=max_prompt_length,
            return_tensors=None,
        )

    def _tokenize_full(prompt: str, response: str):
        msg = (
            f"<|im_start|>system\n{DPO_SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{response}<|im_end|>"
        )
        return tokenizer(
            msg,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors=None,
        )

    batch_prompt = [_tokenize_prompt(p) for p in prompts]
    batch_chosen = [_tokenize_full(p, c) for p, c in zip(prompts, chosen)]
    batch_rejected = [_tokenize_full(p, r) for p, r in zip(prompts, rejected)]

    return Dataset.from_dict({
        "prompt_input_ids": [x["input_ids"] for x in batch_prompt],
        "prompt_attention_mask": [x["attention_mask"] for x in batch_prompt],
        "chosen_input_ids": [x["input_ids"] for x in batch_chosen],
        "chosen_attention_mask": [x["attention_mask"] for x in batch_chosen],
        "rejected_input_ids": [x["input_ids"] for x in batch_rejected],
        "rejected_attention_mask": [x["attention_mask"] for x in batch_rejected],
    })
