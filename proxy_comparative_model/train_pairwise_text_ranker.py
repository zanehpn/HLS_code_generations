#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


DEFAULT_PARETO_METRICS = [
    "Worst-caseLatency",
    "BRAM_18K",
    "LUT",
    "DSP",
    "FF",
]

METRIC_ALIASES = {
    "Worst-caseLatency": [
        "Worst-caseLatency",
        "worst-case-latency",
        "worst_case_latency",
        "latency",
        "Latency",
    ],
    "BRAM_18K": ["BRAM_18K", "bram_18k", "BRAM", "bram"],
    "LUT": ["LUT", "lut"],
    "DSP": ["DSP", "dsp"],
    "FF": ["FF", "ff"],
}


# ------------------------------
# Data structures
# ------------------------------

@dataclass
class DesignSample:
    app: str
    design_id: str
    text: str
    latency: float
    metrics: Dict[str, float]
    anchor_latency: float = 0.0
    anchor_score_target: float = 0.0


@dataclass
class PairSample:
    app: str
    i_idx: int
    j_idx: int
    label: float
    name_i: str
    name_j: str
    latency_i: float
    latency_j: float
    dominance_margin: float = 0.0


class PairTextDataset(Dataset):
    def __init__(self, pairs: List[PairSample]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]


# ------------------------------
# Parsing + pair construction
# ------------------------------

def _extract_metric(meta: Dict, metric_name: str) -> Optional[float]:
    keys = []
    for alias in METRIC_ALIASES.get(metric_name, [metric_name]):
        keys.extend([alias, alias.lower(), alias.upper()])
    seen = set()
    for k in keys:
        if k in seen:
            continue
        seen.add(k)
        if k in meta:
            try:
                return float(meta[k])
            except Exception:
                pass
    return None


def _extract_metrics(meta: Dict, metric_names: List[str]) -> Optional[Dict[str, float]]:
    metrics: Dict[str, float] = {}
    for metric_name in metric_names:
        value = _extract_metric(meta, metric_name)
        if value is None or not math.isfinite(value):
            return None
        metrics[metric_name] = float(value)
    return metrics


def _metric_tuple(sample: DesignSample, metric_names: List[str]) -> Tuple[float, ...]:
    return tuple(float(sample.metrics[m]) for m in metric_names)


def _metric_summary(sample: DesignSample, metric_names: List[str]) -> str:
    return ", ".join(f"{m}={sample.metrics[m]:.4f}" for m in metric_names)


def pareto_compare(metrics_a: Dict[str, float], metrics_b: Dict[str, float], metric_names: List[str]) -> int:
    eps = 1e-12
    a_no_worse = True
    b_no_worse = True
    a_strict = False
    b_strict = False

    for metric_name in metric_names:
        a_val = float(metrics_a[metric_name])
        b_val = float(metrics_b[metric_name])
        if a_val > b_val + eps:
            a_no_worse = False
        if b_val > a_val + eps:
            b_no_worse = False
        if a_val + eps < b_val:
            a_strict = True
        if b_val + eps < a_val:
            b_strict = True

    if a_no_worse and a_strict:
        return -1
    if b_no_worse and b_strict:
        return 1
    return 0


def pareto_relative_gap(metrics_a: Dict[str, float], metrics_b: Dict[str, float], metric_names: List[str]) -> float:
    max_gap = 0.0
    for metric_name in metric_names:
        a_val = float(metrics_a[metric_name])
        b_val = float(metrics_b[metric_name])
        denom = max(abs(a_val), abs(b_val), 1e-9)
        max_gap = max(max_gap, abs(a_val - b_val) / denom)
    return float(max_gap)


def pareto_fronts(samples: List[DesignSample], idxs: List[int], metric_names: List[str]) -> List[List[int]]:
    remaining = list(idxs)
    fronts: List[List[int]] = []

    while remaining:
        front: List[int] = []
        for idx in remaining:
            dominated = False
            for other in remaining:
                if idx == other:
                    continue
                if pareto_compare(samples[other].metrics, samples[idx].metrics, metric_names) == -1:
                    dominated = True
                    break
            if not dominated:
                front.append(idx)

        front = sorted(front, key=lambda x: _metric_tuple(samples[x], metric_names))
        fronts.append(front)
        front_set = set(front)
        remaining = [idx for idx in remaining if idx not in front_set]

    return fronts


def load_designs_from_jsonl(
    path: str,
    pareto_metrics: List[str],
    use_output: bool = True,
) -> List[DesignSample]:
    out: List[DesignSample] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue

            meta = obj.get("metadata", {})
            app = f"{meta.get('source_name', 'unknown')}/{meta.get('algo_name', 'unknown')}"
            design_id = str(meta.get("design_id", f"design_{ln}"))
            metrics = _extract_metrics(meta, pareto_metrics)
            if metrics is None:
                continue
            primary_metric = float(metrics[pareto_metrics[0]])

            input_text = obj.get("input", "")
            output_text = obj.get("output", "") if use_output else ""
            text = f"[APP] {app}\n[DESIGN] {design_id}\n[INPUT]\n{input_text}\n[CODE]\n{output_text}"
            out.append(
                DesignSample(
                    app=app,
                    design_id=design_id,
                    text=text,
                    latency=primary_metric,
                    metrics=metrics,
                )
            )
    return out


def split_by_application_stratified(
    samples: List[DesignSample],
    train_ratio: float,
    seed: int,
    drop_singleton_apps: bool = True,
) -> Tuple[List[int], List[int], Dict[str, List[int]], List[str]]:
    app2idx: Dict[str, List[int]] = {}
    for i, s in enumerate(samples):
        app2idx.setdefault(s.app, []).append(i)

    rng = random.Random(seed)
    train_idx, test_idx = [], []
    dropped_apps: List[str] = []

    for app, idxs in app2idx.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        if len(idxs) == 1:
            if drop_singleton_apps:
                dropped_apps.append(app)
                continue
            train_idx.extend(idxs)
            continue

        n_train = max(1, min(len(idxs) - 1, int(round(len(idxs) * train_ratio))))
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])

    return train_idx, test_idx, app2idx, dropped_apps


def build_pairs(
    samples: List[DesignSample],
    subset_indices: List[int],
    metric_names: List[str],
    mode: str = "disjoint_pairs",
    min_rel_diff: float = 0.05,
    max_pairs_per_app: int = 256,
    bidirectional: bool = True,
    seed: int = 42,
) -> List[PairSample]:
    rng = random.Random(seed)
    app2idx: Dict[str, List[int]] = {}
    for idx in subset_indices:
        app2idx.setdefault(samples[idx].app, []).append(idx)

    pairs: List[PairSample] = []

    for app, idxs in app2idx.items():
        if len(idxs) < 2:
            continue

        comparable_pairs: List[Tuple[int, int, float]] = []
        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                idx_i = idxs[i]
                idx_j = idxs[j]
                relation = pareto_compare(samples[idx_i].metrics, samples[idx_j].metrics, metric_names)
                if relation == 0:
                    continue
                winner, loser = (idx_i, idx_j) if relation == -1 else (idx_j, idx_i)
                gap = pareto_relative_gap(samples[winner].metrics, samples[loser].metrics, metric_names)
                if gap < min_rel_diff:
                    continue
                comparable_pairs.append((winner, loser, gap))

        if not comparable_pairs:
            continue

        raw_pairs: List[Tuple[int, int, float]] = []
        if mode == "best_anchor":
            fronts = pareto_fronts(samples, idxs, metric_names)
            front0 = fronts[0]
            front0_set = set(front0)
            candidates = sorted([idx for idx in idxs if idx not in front0_set], key=lambda x: _metric_tuple(samples[x], metric_names))
            for cand_idx in candidates:
                dominators = [
                    front_idx
                    for front_idx in front0
                    if pareto_compare(samples[front_idx].metrics, samples[cand_idx].metrics, metric_names) == -1
                ]
                if not dominators:
                    continue
                anchor_idx = dominators[0]
                gap = pareto_relative_gap(samples[anchor_idx].metrics, samples[cand_idx].metrics, metric_names)
                if gap < min_rel_diff:
                    continue
                raw_pairs.append((anchor_idx, cand_idx, gap))
        elif mode == "disjoint_pairs":
            shuffled_pairs = list(comparable_pairs)
            rng.shuffle(shuffled_pairs)
            used = set()
            for winner, loser, gap in shuffled_pairs:
                if winner in used or loser in used:
                    continue
                raw_pairs.append((winner, loser, gap))
                used.add(winner)
                used.add(loser)
        else:
            raw_pairs = list(comparable_pairs)

        filtered: List[PairSample] = []
        for i_idx, j_idx, gap in raw_pairs:
            lat_i = samples[i_idx].latency
            lat_j = samples[j_idx].latency
            name_i = f"{samples[i_idx].app}/{samples[i_idx].design_id}"
            name_j = f"{samples[j_idx].app}/{samples[j_idx].design_id}"
            filtered.append(
                PairSample(
                    app=samples[i_idx].app,
                    i_idx=i_idx,
                    j_idx=j_idx,
                    label=1.0,
                    name_i=name_i,
                    name_j=name_j,
                    latency_i=float(lat_i),
                    latency_j=float(lat_j),
                    dominance_margin=float(gap),
                )
            )

            if bidirectional:
                filtered.append(
                    PairSample(
                        app=samples[j_idx].app,
                        i_idx=j_idx,
                        j_idx=i_idx,
                        label=0.0,
                        name_i=name_j,
                        name_j=name_i,
                        latency_i=float(lat_j),
                        latency_j=float(lat_i),
                        dominance_margin=float(gap),
                    )
                )

        if max_pairs_per_app > 0 and len(filtered) > max_pairs_per_app:
            filtered = rng.sample(filtered, max_pairs_per_app)

        pairs.extend(filtered)

    rng.shuffle(pairs)
    return pairs


def _sample_key(sample: DesignSample) -> str:
    return f"{sample.app}/{sample.design_id}"


def latency_pair_target_logit(latency_i: float, latency_j: float, clip_value: float) -> float:
    eps = 1e-9
    lat_i = max(abs(float(latency_i)), eps)
    lat_j = max(abs(float(latency_j)), eps)
    target_logit = math.log(lat_j / lat_i)
    if clip_value > 0:
        target_logit = max(-clip_value, min(clip_value, target_logit))
    return float(target_logit)


def annotate_anchor_relative_targets(samples: List[DesignSample], clip_value: float) -> Dict[str, float]:
    app_to_anchor_latency: Dict[str, float] = {}
    for sample in samples:
        prev = app_to_anchor_latency.get(sample.app)
        if prev is None or sample.latency < prev:
            app_to_anchor_latency[sample.app] = float(sample.latency)

    for sample in samples:
        anchor_latency = float(app_to_anchor_latency[sample.app])
        sample.anchor_latency = anchor_latency
        sample.anchor_score_target = latency_pair_target_logit(
            latency_i=sample.latency,
            latency_j=anchor_latency,
            clip_value=clip_value,
        )

    return app_to_anchor_latency


def anchor_score_to_latency(anchor_latency: float, anchor_score: float) -> float:
    return float(anchor_latency * math.exp(-float(anchor_score)))


def _line_key_from_obj(obj: Dict) -> Optional[str]:
    meta = obj.get("metadata", {}) if isinstance(obj, dict) else {}
    if not isinstance(meta, dict):
        return None
    suite = str(meta.get("source_name", "")).strip()
    algo = str(meta.get("algo_name", "")).strip()
    design_id = str(meta.get("design_id", "")).strip()
    if not suite or not algo or not design_id:
        return None
    return f"{suite}/{algo}/{design_id}"


def export_split_jsonl_by_keys(
    source_jsonl: str,
    train_keys: set,
    test_keys: set,
    out_train_jsonl: str,
    out_test_jsonl: str,
) -> Tuple[int, int]:
    train_cnt = 0
    test_cnt = 0
    with open(source_jsonl, "r", encoding="utf-8") as src,          open(out_train_jsonl, "w", encoding="utf-8") as wt,          open(out_test_jsonl, "w", encoding="utf-8") as ws:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            key = _line_key_from_obj(obj)
            if key is None:
                continue
            if key in train_keys:
                wt.write(raw + "\n")
                train_cnt += 1
            elif key in test_keys:
                ws.write(raw + "\n")
                test_cnt += 1
    return train_cnt, test_cnt


def export_jsonl_by_keys(
    source_jsonl: str,
    keep_keys: set,
    out_jsonl: str,
) -> int:
    count = 0
    with open(source_jsonl, "r", encoding="utf-8") as src, open(out_jsonl, "w", encoding="utf-8") as dst:
        for line in src:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            key = _line_key_from_obj(obj)
            if key is None or key not in keep_keys:
                continue
            dst.write(raw + "\n")
            count += 1
    return count


def export_pairs_jsonl(
    pairs: List[PairSample],
    samples: List[DesignSample],
    out_path: str,
    metric_names: List[str],
) -> int:
    count = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for p in pairs:
            si = samples[p.i_idx]
            sj = samples[p.j_idx]
            obj = {
                "app": p.app,
                "name_i": p.name_i,
                "name_j": p.name_j,
                "design_i": si.design_id,
                "design_j": sj.design_id,
                "latency_i": si.latency,
                "latency_j": sj.latency,
                "metrics_i": {m: float(si.metrics[m]) for m in metric_names},
                "metrics_j": {m: float(sj.metrics[m]) for m in metric_names},
                "pareto_metrics": list(metric_names),
                "dominance_margin": float(p.dominance_margin),
                "label": float(p.label),
            }
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
    return count


# ------------------------------
# Pragma-aware tokenization
# ------------------------------
#
# Register canonical HLS pragma directive heads as atomic special tokens so
# each `#pragma HLS <DIRECTIVE>` becomes a single id instead of 4-6 BPE
# sub-tokens. Texts are preprocessed with `pragma_normalize` before tokenizing
# so the atomic-token matching is reliable across tokenizer implementations.

PRAGMA_TOKEN_MAP: List[Tuple[str, str]] = [
    ("#pragma HLS PIPELINE",             "<pragma_pipeline>"),
    ("#pragma HLS UNROLL",               "<pragma_unroll>"),
    ("#pragma HLS ARRAY_PARTITION",      "<pragma_array_partition>"),
    ("#pragma HLS ARRAY_RESHAPE",        "<pragma_array_reshape>"),
    ("#pragma HLS DATAFLOW",             "<pragma_dataflow>"),
    ("#pragma HLS INLINE",               "<pragma_inline>"),
    ("#pragma HLS INTERFACE",            "<pragma_interface>"),
    ("#pragma HLS LATENCY",              "<pragma_latency>"),
    ("#pragma HLS loop_tripcount",       "<pragma_loop_tripcount>"),
    ("#pragma HLS ALLOCATION",           "<pragma_allocation>"),
    ("#pragma HLS RESOURCE",             "<pragma_resource>"),
    ("#pragma HLS BIND_OP",              "<pragma_bind_op>"),
    ("#pragma HLS BIND_STORAGE",         "<pragma_bind_storage>"),
    ("#pragma HLS STREAM",               "<pragma_stream>"),
    ("#pragma HLS DEPENDENCE",           "<pragma_dependence>"),
    ("#pragma HLS AGGREGATE",            "<pragma_aggregate>"),
    ("#pragma HLS DISAGGREGATE",         "<pragma_disaggregate>"),
    ("#pragma HLS FUNCTION_INSTANTIATE", "<pragma_function_instantiate>"),
    ("#pragma HLS STABLE",               "<pragma_stable>"),
    ("#pragma HLS OCCURRENCE",           "<pragma_occurrence>"),
    ("#pragma HLS EXPRESSION_BALANCE",   "<pragma_expression_balance>"),
    ("#pragma HLS TOP",                  "<pragma_top>"),
    ("#pragma HLS PROTOCOL",             "<pragma_protocol>"),
    ("#pragma HLS PERFORMANCE",          "<pragma_performance>"),
    ("#pragma HLS RESET",                "<pragma_reset>"),
    ("#pragma HLS REWIND",               "<pragma_rewind>"),
]

# Compile patterns once, sorted by source length (longest first) so that
# "ARRAY_PARTITION" never matches inside a shorter directive.
_PRAGMA_COMPILED: List[Tuple[re.Pattern, str]] = [
    (re.compile(re.escape(src), flags=re.IGNORECASE), dst)
    for src, dst in sorted(PRAGMA_TOKEN_MAP, key=lambda p: -len(p[0]))
]


def pragma_normalize(text: str) -> str:
    """Replace canonical HLS pragma headers with atomic placeholder tokens."""
    out = text
    for pat, dst in _PRAGMA_COMPILED:
        out = pat.sub(dst, out)
    return out


def add_pragma_special_tokens(tokenizer, model: Optional[nn.Module] = None) -> int:
    """Register pragma placeholder tokens as atomic special tokens.

    Returns the number of tokens newly added. If a model is provided and new
    tokens were added, its input embedding matrix is resized accordingly.
    """
    new_tokens = [dst for _, dst in PRAGMA_TOKEN_MAP]
    n_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    if model is not None and n_added > 0:
        model.resize_token_embeddings(len(tokenizer))
    return n_added


# ------------------------------
# Pragma-aware attention helpers (legacy, kept for --pragma_attn_bias)
# ------------------------------

def find_pragma_spans(text: str) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    needle = "#pragma HLS"
    i = 0
    n = len(text)
    while i < n:
        idx = text.find(needle, i)
        if idx == -1:
            break
        line_end = text.find("\n", idx)
        if line_end == -1:
            line_end = n
        spans.append((idx, line_end))
        i = line_end + 1
    return spans


def tokenize_with_pragma(
    tokenizer,
    texts: List[str],
    max_length: int,
    with_pragma: bool,
    pragma_tokenization: bool = False,
) -> Dict[str, torch.Tensor]:
    if pragma_tokenization:
        texts = [pragma_normalize(t) for t in texts]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        return_offsets_mapping=with_pragma,
    )
    if not with_pragma:
        return dict(enc)

    offsets = enc.pop("offset_mapping")  # (B, L, 2)
    attn = enc["attention_mask"]
    pragma_mask = torch.zeros_like(attn, dtype=torch.float32)
    for b, text in enumerate(texts):
        spans = find_pragma_spans(text)
        if not spans:
            continue
        for t in range(offsets.size(1)):
            if attn[b, t].item() == 0:
                continue
            s = int(offsets[b, t, 0])
            e = int(offsets[b, t, 1])
            if s == e:
                continue
            for sp_s, sp_e in spans:
                if s < sp_e and e > sp_s:
                    pragma_mask[b, t] = 1.0
                    break
    out = dict(enc)
    out["pragma_mask"] = pragma_mask
    return out


# ------------------------------
# Model
# ------------------------------

class UniXcoderPairRanker(nn.Module):
    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        dropout: float = 0.2,
        pairwise_residual_alpha: float = 0.0,
        freeze_backbone: bool = False,
        pragma_attn_bias: bool = False,
        pragma_attn_bias_init: float = 1.0,
        pragma_bias_per_layer: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pragma_attn_bias = bool(pragma_attn_bias)
        if self.pragma_attn_bias:
            num_layers = self.encoder.config.num_hidden_layers
            if pragma_bias_per_layer:
                self.pragma_bias = nn.Parameter(
                    torch.full((num_layers,), float(pragma_attn_bias_init))
                )
            else:
                self.pragma_bias = nn.Parameter(torch.tensor(float(pragma_attn_bias_init)))
            self.pragma_bias_per_layer = bool(pragma_bias_per_layer)
            if not (hasattr(self.encoder, "embeddings") and hasattr(self.encoder, "encoder")):
                raise RuntimeError(
                    "pragma_attn_bias requires a RoBERTa-family backbone with .embeddings and .encoder"
                )

        self.score_head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden // 2, 1),
        )

        self.pair_head = nn.Sequential(
            nn.Linear(self.hidden * 4, self.hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, self.hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden // 2, 1),
        )

        self.pairwise_residual_alpha = float(pairwise_residual_alpha)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pragma_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.pragma_attn_bias and pragma_mask is not None:
            last = self._encode_with_pragma_bias(input_ids, attention_mask, pragma_mask)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            last = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(last.dtype)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.dropout(pooled)

    def _encode_with_pragma_bias(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pragma_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeds = self.encoder.embeddings(input_ids=input_ids)
        dtype = embeds.dtype
        attn_f = attention_mask.to(dtype)
        pad_bias = (1.0 - attn_f)[:, None, None, :] * torch.finfo(dtype).min
        pragma_f = pragma_mask.to(dtype)[:, None, None, :]

        if self.pragma_bias_per_layer:
            num_layers = self.pragma_bias.shape[0]
            hidden = embeds
            for layer_idx in range(num_layers):
                ext = pad_bias + pragma_f * self.pragma_bias[layer_idx]
                layer_out = self.encoder.encoder.layer[layer_idx](hidden, attention_mask=ext)
                hidden = layer_out[0]
            return hidden
        else:
            ext = pad_bias + pragma_f * self.pragma_bias
            enc_out = self.encoder.encoder(embeds, attention_mask=ext)
            return enc_out.last_hidden_state

    def forward_pair(self, batch_i: Dict[str, torch.Tensor], batch_j: Dict[str, torch.Tensor]):
        h_i = self._encode(
            batch_i["input_ids"],
            batch_i["attention_mask"],
            batch_i.get("pragma_mask"),
        )
        h_j = self._encode(
            batch_j["input_ids"],
            batch_j["attention_mask"],
            batch_j.get("pragma_mask"),
        )

        pred_i = self.score_head(h_i).view(-1)
        pred_j = self.score_head(h_j).view(-1)

        base_logit = pred_i - pred_j
        if self.pairwise_residual_alpha <= 0:
            pair_logit = base_logit
        else:
            diff = h_i - h_j
            abs_diff = torch.abs(diff)
            pair_in = torch.cat([h_i, h_j, diff, abs_diff], dim=-1)
            residual = self.pair_head(pair_in).view(-1)
            pair_logit = base_logit + self.pairwise_residual_alpha * residual

        return pred_i, pred_j, pair_logit


# ------------------------------
# Train/eval helpers
# ------------------------------

def make_collate_fn(
    samples: List[DesignSample],
    tokenizer,
    max_length: int,
    with_pragma: bool = False,
    pragma_tokenization: bool = False,
):
    def collate(batch: List[PairSample]):
        txt_i = [samples[x.i_idx].text for x in batch]
        txt_j = [samples[x.j_idx].text for x in batch]

        tok_i = tokenize_with_pragma(tokenizer, txt_i, max_length, with_pragma, pragma_tokenization)
        tok_j = tokenize_with_pragma(tokenizer, txt_j, max_length, with_pragma, pragma_tokenization)

        labels = torch.tensor([x.label for x in batch], dtype=torch.float32)
        names_i = [x.name_i for x in batch]
        names_j = [x.name_j for x in batch]
        apps = [x.app for x in batch]
        return tok_i, tok_j, labels, names_i, names_j, apps

    return collate


def move_batch(tok: Dict[str, torch.Tensor], device: torch.device):
    return {k: v.to(device) for k, v in tok.items()}


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    pair_loss_weight: float,
    consistency_loss_weight: float,
):
    model.train()
    total_loss = 0.0
    total_pair_bce = 0.0
    total_cons = 0.0
    correct = 0
    total = 0

    for tok_i, tok_j, labels, _, _, _ in loader:
        tok_i = move_batch(tok_i, device)
        tok_j = move_batch(tok_j, device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred_i, pred_j, pair_logit = model.forward_pair(tok_i, tok_j)

        base_logit = pred_i - pred_j
        pair_bce = F.binary_cross_entropy_with_logits(pair_logit, labels)
        cons_mse = F.mse_loss(pair_logit, base_logit)
        loss = (
            pair_loss_weight * pair_bce
            + consistency_loss_weight * cons_mse
        )

        loss.backward()
        optimizer.step()

        bs = labels.numel()
        total += bs
        total_loss += loss.item() * bs
        total_pair_bce += pair_bce.item() * bs
        total_cons += cons_mse.item() * bs
        pred_lbl = (pair_logit > 0).float()
        correct += int((pred_lbl == labels).sum().item())

    return {
        "loss": total_loss / max(1, total),
        "pair_bce": total_pair_bce / max(1, total),
        "consistency_mse": total_cons / max(1, total),
        "label_acc": correct / max(1, total),
        "correct": correct,
        "total": total,
    }


def evaluate(
    model,
    loader,
    device,
    pair_loss_weight: float,
    consistency_loss_weight: float,
):
    model.eval()
    total_loss = 0.0
    total_pair_bce = 0.0
    total_cons = 0.0
    correct = 0
    total = 0

    all_infos = []
    with torch.no_grad():
        for tok_i, tok_j, labels, names_i, names_j, apps in loader:
            tok_i = move_batch(tok_i, device)
            tok_j = move_batch(tok_j, device)
            labels = labels.to(device)

            pred_i, pred_j, pair_logit = model.forward_pair(tok_i, tok_j)
            base_logit = pred_i - pred_j
            pair_bce = F.binary_cross_entropy_with_logits(pair_logit, labels)
            cons_mse = F.mse_loss(pair_logit, base_logit)
            loss = (
                pair_loss_weight * pair_bce
                + consistency_loss_weight * cons_mse
            )

            probs = torch.sigmoid(pair_logit).detach().cpu().tolist()
            pred_lbl = (pair_logit > 0).float()
            bs = labels.numel()

            total += bs
            total_loss += loss.item() * bs
            total_pair_bce += pair_bce.item() * bs
            total_cons += cons_mse.item() * bs
            correct += int((pred_lbl == labels).sum().item())

            lbls = labels.detach().cpu().tolist()
            for i in range(bs):
                all_infos.append(
                    {
                        "app": apps[i],
                        "name_i": names_i[i],
                        "name_j": names_j[i],
                        "label": float(lbls[i]),
                        "prob_i_better": float(probs[i]),
                    }
                )

    return {
        "loss": total_loss / max(1, total),
        "pair_bce": total_pair_bce / max(1, total),
        "consistency_mse": total_cons / max(1, total),
        "label_acc": correct / max(1, total),
        "correct": correct,
        "total": total,
        "pair_infos": all_infos,
    }


def best_anchor_pairwise_report(
    model,
    tokenizer,
    samples: List[DesignSample],
    test_indices: List[int],
    metric_names: List[str],
    device: torch.device,
    max_length: int,
    max_apps: int = 20,
    pragma_tokenization: bool = False,
):
    model.eval()
    app2idx: Dict[str, List[int]] = {}
    for idx in test_indices:
        app2idx.setdefault(samples[idx].app, []).append(idx)

    shown = 0
    print("\nBest-anchor pairwise report (test split):")
    with torch.no_grad():
        for app, idxs in sorted(app2idx.items()):
            if len(idxs) < 2:
                continue
            fronts = pareto_fronts(samples, idxs, metric_names)
            front0 = fronts[0]
            front0_set = set(front0)
            rows_meta: List[Tuple[int, int]] = []
            for cand_idx in sorted([idx for idx in idxs if idx not in front0_set], key=lambda x: _metric_tuple(samples[x], metric_names)):
                dominators = [
                    front_idx
                    for front_idx in front0
                    if pareto_compare(samples[front_idx].metrics, samples[cand_idx].metrics, metric_names) == -1
                ]
                if dominators:
                    rows_meta.append((cand_idx, dominators[0]))

            if not rows_meta:
                continue

            with_pragma = getattr(model, "pragma_attn_bias", False)
            tok_anchor = tokenize_with_pragma(
                tokenizer,
                [samples[anchor_idx].text for _, anchor_idx in rows_meta],
                max_length,
                with_pragma,
                pragma_tokenization,
            )
            tok_cand = tokenize_with_pragma(
                tokenizer,
                [samples[cand_idx].text for cand_idx, _ in rows_meta],
                max_length,
                with_pragma,
                pragma_tokenization,
            )
            tok_anchor = move_batch(tok_anchor, device)
            tok_cand = move_batch(tok_cand, device)

            _, _, logit = model.forward_pair(tok_cand, tok_anchor)
            probs = torch.sigmoid(logit).detach().cpu().tolist()  # P(candidate better than anchor)

            rows = []
            for (cand_idx, anchor_idx), prob in zip(rows_meta, probs):
                rows.append(
                    (
                        samples[cand_idx].design_id,
                        samples[anchor_idx].design_id,
                        _metric_summary(samples[cand_idx], metric_names),
                        _metric_summary(samples[anchor_idx], metric_names),
                        prob,
                    )
                )
            rows.sort(key=lambda x: x[4], reverse=True)

            print(
                f"  [{app}] pareto_front={len(front0)}, dominated_candidates={len(rows_meta)}"
            )
            for cand_id, anchor_id, cand_metrics, anchor_metrics, prob in rows[:5]:
                print(
                    f"    cand={cand_id} [{cand_metrics}] | anchor={anchor_id} [{anchor_metrics}] | "
                    f"P(cand>anchor)={prob:.4f}, gt=0"
                )

            shown += 1
            if shown >= max_apps:
                break


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_input_mode(args) -> str:
    has_single_file = bool(args.metrics_jsonl)
    has_explicit_split = bool(args.train_metrics_jsonl or args.test_metrics_jsonl)

    if has_single_file and has_explicit_split:
        raise RuntimeError(
            "Use either --metrics_jsonl or both --train_metrics_jsonl and --test_metrics_jsonl, not both modes together."
        )
    if has_single_file:
        return "single"
    if args.train_metrics_jsonl and args.test_metrics_jsonl:
        return "explicit_split"
    if has_explicit_split:
        raise RuntimeError("Explicit split mode requires both --train_metrics_jsonl and --test_metrics_jsonl.")
    raise RuntimeError("Missing input data. Provide --metrics_jsonl, or provide both --train_metrics_jsonl and --test_metrics_jsonl.")


def combine_split_samples(
    train_samples: List[DesignSample],
    test_samples: List[DesignSample],
) -> Tuple[List[DesignSample], List[int], List[int]]:
    samples = list(train_samples)
    train_idx = list(range(len(train_samples)))
    test_offset = len(train_samples)
    samples.extend(test_samples)
    test_idx = list(range(test_offset, test_offset + len(test_samples)))
    return samples, train_idx, test_idx


# ------------------------------
# Main
# ------------------------------

def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    args.pareto_metrics = [m.strip() for m in args.pareto_metrics if str(m).strip()]
    if not args.pareto_metrics:
        raise RuntimeError("pareto_metrics must contain at least one metric name.")
    deduped_metrics = []
    seen_metrics = set()
    for metric_name in args.pareto_metrics:
        if metric_name in seen_metrics:
            continue
        seen_metrics.add(metric_name)
        deduped_metrics.append(metric_name)
    args.pareto_metrics = deduped_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"Pareto metrics: {', '.join(args.pareto_metrics)}")
    if args.target_metric:
        print("Note: pure pairwise mode ignores --target_metric and uses --pareto_metrics instead.")
    input_mode = resolve_input_mode(args)
    train_source_jsonl = args.metrics_jsonl
    test_source_jsonl = args.metrics_jsonl

    if input_mode == "single":
        print(f"Loading dataset: {args.metrics_jsonl}")
        samples = load_designs_from_jsonl(
            args.metrics_jsonl,
            pareto_metrics=args.pareto_metrics,
            use_output=(not args.no_use_output),
        )
        train_idx, test_idx, _, dropped_apps = split_by_application_stratified(
            samples,
            args.train_ratio,
            args.seed,
            drop_singleton_apps=args.drop_singleton_apps,
        )
    else:
        train_source_jsonl = args.train_metrics_jsonl
        test_source_jsonl = args.test_metrics_jsonl
        print(f"Loading explicit train dataset: {train_source_jsonl}")
        train_samples = load_designs_from_jsonl(
            train_source_jsonl,
            pareto_metrics=args.pareto_metrics,
            use_output=(not args.no_use_output),
        )
        print(f"Loading explicit test dataset: {test_source_jsonl}")
        test_samples = load_designs_from_jsonl(
            test_source_jsonl,
            pareto_metrics=args.pareto_metrics,
            use_output=(not args.no_use_output),
        )
        train_keys = {_sample_key(sample) for sample in train_samples}
        test_keys = {_sample_key(sample) for sample in test_samples}
        overlap_keys = train_keys & test_keys
        if overlap_keys:
            raise RuntimeError(
                f"Explicit train/test files overlap on {len(overlap_keys)} designs. "
                "Please de-duplicate the split inputs."
            )
        samples, train_idx, test_idx = combine_split_samples(train_samples, test_samples)
        dropped_apps = []
        print("Explicit split mode: --train_ratio and singleton-app split logic are ignored.")

    print(f"Loaded designs: {len(samples)}")
    if len(samples) < 10:
        raise RuntimeError("Too few samples after loading.")
    if args.point_loss_weight != 0.0 or args.target_gap_loss_weight != 0.0:
        print(
            "Note: pure pairwise mode ignores --point_loss_weight and --target_gap_loss_weight."
        )
    if args.target_logit_clip > 0.0:
        print("Note: pure pairwise mode ignores --target_logit_clip.")
    train_apps = {samples[i].app for i in train_idx}
    test_apps = {samples[i].app for i in test_idx}
    common_apps = train_apps & test_apps
    train_only_apps = train_apps - test_apps
    test_only_apps = test_apps - train_apps

    print(f"Train designs: {len(train_idx)} | Test designs: {len(test_idx)}")
    print(
        f"App coverage: train={len(train_apps)}, test={len(test_apps)}, both={len(common_apps)}, "
        f"train_only={len(train_only_apps)}, test_only={len(test_only_apps)}"
    )
    if dropped_apps:
        print(f"Dropped singleton apps: {len(dropped_apps)}")

    if args.require_app_overlap and (len(train_only_apps) > 0 or len(test_only_apps) > 0):
        raise RuntimeError(
            "Application overlap requirement failed. "
            f"train_only={len(train_only_apps)}, test_only={len(test_only_apps)}"
        )

    train_pairs = build_pairs(
        samples,
        train_idx,
        metric_names=args.pareto_metrics,
        mode=args.pair_selection_mode,
        min_rel_diff=args.pair_min_rel_diff,
        max_pairs_per_app=args.pair_max_per_app_train,
        bidirectional=(not args.no_bidirectional_pairs),
        seed=args.seed,
    )
    test_pairs = build_pairs(
        samples,
        test_idx,
        metric_names=args.pareto_metrics,
        mode=args.pair_selection_mode,
        min_rel_diff=args.pair_min_rel_diff,
        max_pairs_per_app=args.pair_max_per_app_test,
        bidirectional=(not args.no_bidirectional_pairs),
        seed=args.seed + 1,
    )

    print(f"Train pairs: {len(train_pairs)} | Test pairs: {len(test_pairs)}")
    if len(train_pairs) == 0 or len(test_pairs) == 0:
        raise RuntimeError("No pairs built. Try lower pair_min_rel_diff or increase data.")

    split_dir = args.split_output_dir
    if split_dir:
        os.makedirs(split_dir, exist_ok=True)
        split_tag = args.split_tag if args.split_tag else f"pair_text_ranker_{args.pair_selection_mode}"
        split_train_jsonl = os.path.join(split_dir, f"{split_tag}_train.jsonl")
        split_test_jsonl = os.path.join(split_dir, f"{split_tag}_test.jsonl")
        split_train_pairs = os.path.join(split_dir, f"{split_tag}_train_pairs.jsonl")
        split_test_pairs = os.path.join(split_dir, f"{split_tag}_test_pairs.jsonl")

        train_keys = {_sample_key(samples[i]) for i in train_idx}
        test_keys = {_sample_key(samples[i]) for i in test_idx}

        if input_mode == "single":
            tr_cnt, te_cnt = export_split_jsonl_by_keys(
                args.metrics_jsonl,
                train_keys,
                test_keys,
                split_train_jsonl,
                split_test_jsonl,
            )
        else:
            tr_cnt = export_jsonl_by_keys(train_source_jsonl, train_keys, split_train_jsonl)
            te_cnt = export_jsonl_by_keys(test_source_jsonl, test_keys, split_test_jsonl)
        tr_pair_cnt = export_pairs_jsonl(train_pairs, samples, split_train_pairs, args.pareto_metrics)
        te_pair_cnt = export_pairs_jsonl(test_pairs, samples, split_test_pairs, args.pareto_metrics)

        print("Split artifacts written:")
        print(f"  train jsonl: {split_train_jsonl} ({tr_cnt})")
        print(f"  test  jsonl: {split_test_jsonl} ({te_cnt})")
        print(f"  train pairs: {split_train_pairs} ({tr_pair_cnt})")
        print(f"  test  pairs: {split_test_pairs} ({te_pair_cnt})")

        if args.export_splits_only:
            print("Export-only mode: skip training.")
            return

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    n_pragma_added = 0
    if args.pragma_tokenization:
        n_pragma_added = add_pragma_special_tokens(tokenizer, model=None)
        print(
            f"Pragma-aware tokenization ENABLED "
            f"(added {n_pragma_added}/{len(PRAGMA_TOKEN_MAP)} atomic pragma tokens; "
            f"vocab={len(tokenizer)})"
        )

    train_ds = PairTextDataset(train_pairs)
    test_ds = PairTextDataset(test_pairs)
    collate_fn = make_collate_fn(
        samples,
        tokenizer,
        args.max_length,
        with_pragma=args.pragma_attn_bias,
        pragma_tokenization=args.pragma_tokenization,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )

    model = UniXcoderPairRanker(
        model_name=args.model_name,
        dropout=args.dropout,
        pairwise_residual_alpha=args.pairwise_residual_alpha,
        freeze_backbone=args.freeze_backbone,
        pragma_attn_bias=args.pragma_attn_bias,
        pragma_attn_bias_init=args.pragma_attn_bias_init,
        pragma_bias_per_layer=args.pragma_bias_per_layer,
    ).to(device)
    if args.pragma_tokenization and n_pragma_added > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))
        # Move the newly initialized embedding rows to the training device.
        model.to(device)
    if args.pragma_attn_bias:
        print(
            f"Pragma-aware attention bias ENABLED "
            f"(init={args.pragma_attn_bias_init}, per_layer={args.pragma_bias_per_layer})"
        )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: trainable={trainable:,} / total={total:,}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)

    best_acc = -1.0
    best_path = os.path.join(args.output_dir, "best_pair_text_ranker.pt")
    history = {
        "train_loss": [],
        "train_pair_bce": [],
        "train_consistency_mse": [],
        "train_label_acc": [],
        "test_loss": [],
        "test_pair_bce": [],
        "test_consistency_mse": [],
        "test_label_acc": [],
        "lr": [],
    }
    history_path = os.path.join(args.output_dir, "history.json")

    for epoch in range(args.epochs):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            pair_loss_weight=args.pair_loss_weight,
            consistency_loss_weight=args.consistency_loss_weight,
        )
        te = evaluate(
            model,
            test_loader,
            device,
            pair_loss_weight=args.pair_loss_weight,
            consistency_loss_weight=args.consistency_loss_weight,
        )
        history["train_loss"].append(float(tr["loss"]))
        history["train_pair_bce"].append(float(tr["pair_bce"]))
        history["train_consistency_mse"].append(float(tr["consistency_mse"]))
        history["train_label_acc"].append(float(tr["label_acc"]))
        history["test_loss"].append(float(te["loss"]))
        history["test_pair_bce"].append(float(te["pair_bce"]))
        history["test_consistency_mse"].append(float(te["consistency_mse"]))
        history["test_label_acc"].append(float(te["label_acc"]))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        print(f"Epoch {epoch:03d}:")
        print(
            f"  Train - Label-ACC: {tr['label_acc']:.4f} ({tr['correct']}/{tr['total']}), "
            f"Loss: {tr['loss']:.4f}, Pair-BCE: {tr['pair_bce']:.4f}, "
            f"Cons-MSE: {tr['consistency_mse']:.4f}"
        )
        print(
            f"  Test  - Label-ACC: {te['label_acc']:.4f} ({te['correct']}/{te['total']}), "
            f"Loss: {te['loss']:.4f}, Pair-BCE: {te['pair_bce']:.4f}, "
            f"Cons-MSE: {te['consistency_mse']:.4f}"
        )
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        if te["label_acc"] > best_acc:
            best_acc = te["label_acc"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "best_test_label_acc": best_acc,
                },
                best_path,
            )
            print(f"  Saved best checkpoint: {best_path}")

        if (epoch + 1) % args.lr_step == 0:
            scheduler.step()

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("\nTraining done.")
    print(f"Best Test Label-ACC: {best_acc:.4f}")
    print(f"History saved to: {history_path}")

    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)

    best_anchor_pairwise_report(
        model,
        tokenizer,
        samples,
        test_idx,
        args.pareto_metrics,
        device,
        max_length=args.max_length,
        max_apps=args.report_max_apps,
        pragma_tokenization=args.pragma_tokenization,
    )


def build_argparser():
    p = argparse.ArgumentParser(description="Train UniXcoder pure pairwise text reranker for HLS design comparison")
    p.add_argument("--metrics_jsonl", type=str,
                   help="Single jsonl input; the script will split it into train/test internally")
    p.add_argument("--train_metrics_jsonl", type=str,
                   help="Explicit train jsonl input; use together with --test_metrics_jsonl")
    p.add_argument("--test_metrics_jsonl", type=str,
                   help="Explicit test jsonl input; use together with --train_metrics_jsonl")
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--model_name", type=str, default="microsoft/unixcoder-base")
    p.add_argument("--target_metric", type=str, default="latency",
                   help="Compatibility-only argument; Pareto comparison uses --pareto_metrics")
    p.add_argument("--pareto_metrics", type=str, nargs="+", default=list(DEFAULT_PARETO_METRICS),
                   help="Metric names jointly minimized for Pareto dominance comparison")

    p.add_argument("--train_ratio", type=float, default=0.8,
                   help="Train split ratio (default 0.8 => 8:2)")
    p.add_argument("--pair_selection_mode", type=str, default="disjoint_pairs", choices=["disjoint_pairs", "best_anchor", "all_pairs"],
                   help="Pair construction mode; disjoint_pairs=random non-overlapping comparable pairs, best_anchor=compare each dominated design against a Pareto-front anchor, all_pairs=all Pareto-comparable pairs")
    p.add_argument("--pair_min_rel_diff", type=float, default=0.10,
                   help="Minimum relative gap across Pareto metrics for keeping a comparable pair")
    p.add_argument("--pair_max_per_app_train", type=int, default=256)
    p.add_argument("--pair_max_per_app_test", type=int, default=256)
    p.add_argument("--no_bidirectional_pairs", action="store_true")
    p.add_argument("--drop_singleton_apps", action="store_true", default=True,
                   help="Drop apps with <2 designs so each kept app can appear in both train and test")
    p.add_argument("--keep_singleton_apps", dest="drop_singleton_apps", action="store_false",
                   help="Keep singleton apps (they can only appear in one split)")
    p.add_argument("--require_app_overlap", action="store_true", default=True,
                   help="Require no train-only/test-only apps after split")
    p.add_argument("--allow_app_nonoverlap", dest="require_app_overlap", action="store_false",
                   help="Allow train-only/test-only apps")
    p.add_argument("--split_output_dir", type=str,
                   default=str(Path(__file__).resolve().parents[2] / "GNN/dataset"),
                   help="Directory for exported split jsonl and pair files")
    p.add_argument("--split_tag", type=str, default="pair_text_ranker_8_2_disjoint",
                   help="Prefix for exported split artifacts")
    p.add_argument("--export_splits_only", action="store_true",
                   help="Only export split jsonl/pairs, skip model training")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_gamma", type=float, default=0.9)
    p.add_argument("--lr_step", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.2)

    p.add_argument("--pair_loss_weight", type=float, default=1.0)
    p.add_argument("--point_loss_weight", type=float, default=0.5,
                   help="Compatibility-only argument; ignored in pure pairwise mode")
    p.add_argument("--consistency_loss_weight", type=float, default=0.5)
    p.add_argument("--target_gap_loss_weight", type=float, default=0.1,
                   help="Compatibility-only argument; ignored in pure pairwise mode")
    p.add_argument("--pairwise_residual_alpha", type=float, default=0.0)
    p.add_argument("--pragma_tokenization", action="store_true",
                   help="Enable pragma-aware tokenization: register each #pragma HLS "
                        "directive (PIPELINE/UNROLL/ARRAY_PARTITION/...) as an atomic "
                        "special token and resize the model embedding matrix accordingly")
    p.add_argument("--pragma_attn_bias", action="store_true",
                   help="Enable learned additive attention bias on #pragma HLS tokens")
    p.add_argument("--pragma_attn_bias_init", type=float, default=1.0,
                   help="Initial value for the learnable pragma attention bias")
    p.add_argument("--pragma_bias_per_layer", action="store_true",
                   help="Use a separate learnable pragma bias per encoder layer")
    p.add_argument("--target_logit_clip", type=float, default=8.0,
                   help="Compatibility-only argument; ignored in pure pairwise mode")

    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--no_use_output", action="store_true", help="Only use input text, not output code")

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--pin_memory", action="store_true")

    p.add_argument("--report_max_apps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
