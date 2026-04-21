import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


LATENCY_ALIASES = ["latency", "Latency", "cycle", "cycles", "lat", "Worst-caseLatency", "Best-caseLatency"]


@dataclass
class DesignSample:
    application: str
    design_id: str
    text: str
    metric: float


@dataclass
class PairSample:
    application: str
    left_id: str
    right_id: str
    left_text: str
    right_text: str
    left_metric: float
    right_metric: float
    label: float


class PairDataset(Dataset):
    def __init__(self, pairs: Sequence[PairSample]):
        self.pairs = list(pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> PairSample:
        return self.pairs[idx]


class MetricTransform:
    def __init__(self, values: Sequence[float]):
        xs = torch.tensor([max(float(v), 0.0) for v in values], dtype=torch.float32)
        xs = torch.log1p(xs)
        self.mean = float(xs.mean().item())
        self.std = float(xs.std(unbiased=False).item())
        if self.std < 1e-6:
            self.std = 1.0

    def encode(self, value: float) -> float:
        return (math.log1p(max(float(value), 0.0)) - self.mean) / self.std


class UnixCoderHybridRanker(nn.Module):
    def __init__(
        self,
        model_name: str,
        dropout: float = 0.1,
        pairwise_residual_alpha: float = 0.1,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.pointwise_decoder = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.pairwise_decoder = nn.Sequential(
            nn.Linear(hidden * 4, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )
        self.pairwise_residual_alpha = float(pairwise_residual_alpha)

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False

    def encode_text(self, token_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        outputs = self.encoder(**token_batch)
        hidden = outputs.last_hidden_state
        mask = token_batch["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.dropout(pooled)

    def forward_once(self, token_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        emb = self.encode_text(token_batch)
        pred = self.pointwise_decoder(emb).view(-1)
        return pred, emb

    def forward_pair(
        self,
        left_batch: Dict[str, torch.Tensor],
        right_batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pred_left, emb_left = self.forward_once(left_batch)
        pred_right, emb_right = self.forward_once(right_batch)
        diff = emb_left - emb_right
        abs_diff = torch.abs(diff)
        base_logit = pred_left - pred_right
        residual = self.pairwise_decoder(torch.cat([emb_left, emb_right, diff, abs_diff], dim=-1)).view(-1)
        pair_logit = base_logit + self.pairwise_residual_alpha * residual
        return pred_left, pred_right, pair_logit


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metadata_view(row: dict) -> dict:
    value = row.get("metadata")
    return value if isinstance(value, dict) else {}


def first_non_empty(row: dict, keys: Sequence[str]) -> Optional[str]:
    md = metadata_view(row)
    for key in keys:
        for src in (row, md):
            value = src.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def infer_application(row: dict) -> Optional[str]:
    direct = first_non_empty(
        row,
        ["application", "app", "app_name", "kernel_name", "algo_name", "benchmark_name"],
    )
    if direct and "/" in direct:
        return direct

    suite = first_non_empty(row, ["suite", "suite_name", "benchmark_suite", "source_name"])
    algo = direct
    if suite and algo:
        return f"{suite}/{algo}"

    path = first_non_empty(row, ["file_path", "source_path", "code_path", "ll_path"])
    if path:
        parts = path.replace("\\", "/").split("/")
        if "design_package" in parts:
            idx = parts.index("design_package")
            if len(parts) >= idx + 4:
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
        if "design_package_src_only" in parts:
            idx = parts.index("design_package_src_only")
            if len(parts) >= idx + 4:
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
        kernel_positions = [i for i, token in enumerate(parts) if token == "kernels"]
        if kernel_positions:
            idx = kernel_positions[-1]
            if len(parts) >= idx + 3:
                return f"{parts[idx + 1]}/{parts[idx + 2]}"
    return direct


def infer_design_id(row: dict) -> Optional[str]:
    value = first_non_empty(row, ["design_id", "design", "variant", "id"])
    if value:
        return value
    path = first_non_empty(row, ["file_path", "source_path", "code_path", "ll_path"])
    if path:
        for token in reversed(path.replace("\\", "/").split("/")):
            if token.startswith("design_"):
                return token
    return None


def infer_code_text(row: dict, use_output: bool) -> Optional[str]:
    code_keys = [
        "code",
        "combined_code",
        "candidate_code",
        "source",
        "source_code",
        "c_code",
        "text",
    ]
    output_keys = ["output", "completion", "response", "generated_code"]
    input_keys = ["input", "prompt"]
    if use_output:
        text = first_non_empty(row, output_keys)
        if text:
            return text
    text = first_non_empty(row, code_keys)
    if text:
        return text
    text = first_non_empty(row, output_keys)
    if text:
        return text
    return first_non_empty(row, input_keys)


def extract_metric(row: dict, target_metric: str) -> Optional[float]:
    candidates = [target_metric, target_metric.lower(), target_metric.upper()]
    aliases = {
        "latency": LATENCY_ALIASES,
        "lut": ["lut", "LUT", "lut_count"],
        "dsp": ["dsp", "DSP", "dsp_count"],
        "bram": ["bram", "BRAM", "BRAM_18K", "bram_18k"],
        "bram_18k": ["bram", "BRAM", "BRAM_18K", "bram_18k"],
    }
    for alias in aliases.get(target_metric.lower(), []):
        if alias not in candidates:
            candidates.append(alias)

    md = metadata_view(row)
    for src in (row, md):
        for key in candidates:
            if key in src:
                try:
                    return float(src[key])
                except Exception:
                    pass
    metrics = row.get("metrics")
    if isinstance(metrics, dict):
        for key in candidates:
            if key in metrics:
                try:
                    return float(metrics[key])
                except Exception:
                    pass
    return None


def load_designs_from_jsonl(path: str, target_metric: str, use_output: bool) -> List[DesignSample]:
    rows = load_jsonl(path)
    samples: List[DesignSample] = []
    skipped = 0
    for row in rows:
        application = infer_application(row)
        design_id = infer_design_id(row)
        text = infer_code_text(row, use_output=use_output)
        metric = extract_metric(row, target_metric)
        if not application or not design_id or not text or metric is None:
            skipped += 1
            continue
        samples.append(
            DesignSample(
                application=application,
                design_id=design_id,
                text=text,
                metric=float(metric),
            )
        )
    print(f"Loaded {len(samples)} usable designs from {path}; skipped {skipped}")
    return samples


def split_by_application(samples: Sequence[DesignSample], train_ratio: float, seed: int) -> Tuple[List[DesignSample], List[DesignSample]]:
    by_app: Dict[str, List[DesignSample]] = defaultdict(list)
    for sample in samples:
        by_app[sample.application].append(sample)

    rng = random.Random(seed)
    train_samples: List[DesignSample] = []
    test_samples: List[DesignSample] = []

    for _app, app_samples in by_app.items():
        app_samples = list(app_samples)
        rng.shuffle(app_samples)
        n = len(app_samples)
        if n == 1:
            train_samples.extend(app_samples)
            continue
        split = int(round(n * train_ratio))
        split = max(1, min(n - 1, split))
        train_samples.extend(app_samples[:split])
        test_samples.extend(app_samples[split:])
    return train_samples, test_samples


def build_best_anchor_pairs(
    samples: Sequence[DesignSample],
    min_rel_diff: float,
    bidirectional: bool,
) -> List[PairSample]:
    by_app: Dict[str, List[DesignSample]] = defaultdict(list)
    for sample in samples:
        by_app[sample.application].append(sample)

    pairs: List[PairSample] = []
    skipped_singleton = 0
    skipped_close = 0

    for app, app_samples in by_app.items():
        if len(app_samples) < 2:
            skipped_singleton += 1
            continue
        anchor = min(app_samples, key=lambda item: item.metric)
        for sample in app_samples:
            if sample.design_id == anchor.design_id:
                continue
            rel_diff = abs(sample.metric - anchor.metric) / max(abs(anchor.metric), 1e-8)
            if rel_diff < min_rel_diff:
                skipped_close += 1
                continue
            pairs.append(
                PairSample(
                    application=app,
                    left_id=sample.design_id,
                    right_id=anchor.design_id,
                    left_text=sample.text,
                    right_text=anchor.text,
                    left_metric=float(sample.metric),
                    right_metric=float(anchor.metric),
                    label=1.0 if sample.metric < anchor.metric else 0.0,
                )
            )
            if bidirectional:
                pairs.append(
                    PairSample(
                        application=app,
                        left_id=anchor.design_id,
                        right_id=sample.design_id,
                        left_text=anchor.text,
                        right_text=sample.text,
                        left_metric=float(anchor.metric),
                        right_metric=float(sample.metric),
                        label=1.0 if anchor.metric < sample.metric else 0.0,
                    )
                )

    print(
        "Built best-anchor pairs: "
        f"{len(pairs)} (bidirectional={bidirectional}, skipped_singleton_apps={skipped_singleton}, skipped_close={skipped_close})"
    )
    return pairs


def collate_pairs(batch: Sequence[PairSample], tokenizer, max_length: int, transform: MetricTransform):
    left_texts = [item.left_text for item in batch]
    right_texts = [item.right_text for item in batch]

    left_tok = tokenizer(
        left_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    right_tok = tokenizer(
        right_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    left_targets = torch.tensor([transform.encode(item.left_metric) for item in batch], dtype=torch.float32)
    right_targets = torch.tensor([transform.encode(item.right_metric) for item in batch], dtype=torch.float32)
    labels = torch.tensor([item.label for item in batch], dtype=torch.float32)
    meta = [
        {
            "application": item.application,
            "left_id": item.left_id,
            "right_id": item.right_id,
            "left_metric": item.left_metric,
            "right_metric": item.right_metric,
        }
        for item in batch
    ]
    return left_tok, right_tok, left_targets, right_targets, labels, meta


def move_token_batch_to_device(token_batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in token_batch.items()}


def train_one_epoch(
    model: UnixCoderHybridRanker,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    point_loss_weight: float,
    pair_loss_weight: float,
    consistency_loss_weight: float,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_point = 0.0
    total_pair = 0.0
    total_cons = 0.0
    total_correct = 0
    total_examples = 0

    for left_tok, right_tok, left_targets, right_targets, labels, _meta in loader:
        left_tok = move_token_batch_to_device(left_tok, device)
        right_tok = move_token_batch_to_device(right_tok, device)
        left_targets = left_targets.to(device)
        right_targets = right_targets.to(device)
        labels = labels.to(device)

        pred_left, pred_right, pair_logit = model.forward_pair(left_tok, right_tok)
        point_loss = F.mse_loss(pred_left, left_targets) + F.mse_loss(pred_right, right_targets)
        pair_loss = F.binary_cross_entropy_with_logits(pair_logit, labels)
        consistency_loss = F.mse_loss(pair_logit, pred_left - pred_right)
        loss = (
            point_loss_weight * point_loss
            + pair_loss_weight * pair_loss
            + consistency_loss_weight * consistency_loss
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_point += float(point_loss.item()) * batch_size
        total_pair += float(pair_loss.item()) * batch_size
        total_cons += float(consistency_loss.item()) * batch_size
        total_correct += int(((pair_logit > 0).float() == labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "point_mse": total_point / max(total_examples, 1),
        "pair_bce": total_pair / max(total_examples, 1),
        "consistency_mse": total_cons / max(total_examples, 1),
        "label_acc": total_correct / max(total_examples, 1),
        "correct": total_correct,
        "num_pairs": total_examples,
    }


@torch.no_grad()
def evaluate(
    model: UnixCoderHybridRanker,
    loader: DataLoader,
    device: torch.device,
    point_loss_weight: float,
    pair_loss_weight: float,
    consistency_loss_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_point = 0.0
    total_pair = 0.0
    total_cons = 0.0
    total_correct = 0
    total_examples = 0

    for left_tok, right_tok, left_targets, right_targets, labels, _meta in loader:
        left_tok = move_token_batch_to_device(left_tok, device)
        right_tok = move_token_batch_to_device(right_tok, device)
        left_targets = left_targets.to(device)
        right_targets = right_targets.to(device)
        labels = labels.to(device)

        pred_left, pred_right, pair_logit = model.forward_pair(left_tok, right_tok)
        point_loss = F.mse_loss(pred_left, left_targets) + F.mse_loss(pred_right, right_targets)
        pair_loss = F.binary_cross_entropy_with_logits(pair_logit, labels)
        consistency_loss = F.mse_loss(pair_logit, pred_left - pred_right)
        loss = (
            point_loss_weight * point_loss
            + pair_loss_weight * pair_loss
            + consistency_loss_weight * consistency_loss
        )

        batch_size = labels.numel()
        total_loss += float(loss.item()) * batch_size
        total_point += float(point_loss.item()) * batch_size
        total_pair += float(pair_loss.item()) * batch_size
        total_cons += float(consistency_loss.item()) * batch_size
        total_correct += int(((pair_logit > 0).float() == labels).sum().item())
        total_examples += batch_size

    return {
        "loss": total_loss / max(total_examples, 1),
        "point_mse": total_point / max(total_examples, 1),
        "pair_bce": total_pair / max(total_examples, 1),
        "consistency_mse": total_cons / max(total_examples, 1),
        "label_acc": total_correct / max(total_examples, 1),
        "correct": total_correct,
        "num_pairs": total_examples,
    }


def save_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UnixCoder hybrid pointwise+pairwise ranker with best-anchor pairs only.")
    parser.add_argument("--metrics_jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="microsoft/unixcoder-base")
    parser.add_argument("--target_metric", default="latency")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pairwise_residual_alpha", type=float, default=0.1)
    parser.add_argument("--point_loss_weight", type=float, default=1.0)
    parser.add_argument("--pair_loss_weight", type=float, default=1.0)
    parser.add_argument("--consistency_loss_weight", type=float, default=0.1)
    parser.add_argument("--pair_min_rel_diff", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--use_output", action="store_true")
    parser.add_argument("--no_bidirectional_pairs", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("Training UnixCoder hybrid best-anchor ranker")
    print("=" * 80)
    print(f"device: {device}")

    samples = load_designs_from_jsonl(
        path=args.metrics_jsonl,
        target_metric=args.target_metric,
        use_output=args.use_output,
    )
    train_samples, test_samples = split_by_application(samples, train_ratio=args.train_ratio, seed=args.seed)
    print(f"train designs: {len(train_samples)}")
    print(f"test designs : {len(test_samples)}")

    train_pairs = build_best_anchor_pairs(
        train_samples,
        min_rel_diff=args.pair_min_rel_diff,
        bidirectional=not args.no_bidirectional_pairs,
    )
    test_pairs = build_best_anchor_pairs(
        test_samples,
        min_rel_diff=args.pair_min_rel_diff,
        bidirectional=not args.no_bidirectional_pairs,
    )
    if not train_pairs:
        raise RuntimeError("No training pairs were built. Check split settings or pair_min_rel_diff.")

    transform = MetricTransform([sample.metric for sample in train_samples])
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(
        PairDataset(train_pairs),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer, args.max_length, transform),
    )
    test_loader = DataLoader(
        PairDataset(test_pairs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda batch: collate_pairs(batch, tokenizer, args.max_length, transform),
    )

    model = UnixCoderHybridRanker(
        model_name=args.model_name,
        dropout=args.dropout,
        pairwise_residual_alpha=args.pairwise_residual_alpha,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = {
        "train_loss": [],
        "train_point_mse": [],
        "train_pair_bce": [],
        "train_consistency_mse": [],
        "train_label_acc": [],
        "test_loss": [],
        "test_point_mse": [],
        "test_pair_bce": [],
        "test_consistency_mse": [],
        "test_label_acc": [],
    }

    best_acc = -1.0
    best_ckpt = os.path.join(args.output_dir, "best_unixcoder_hybrid_best_anchor.pt")
    history_path = os.path.join(args.output_dir, "history.json")

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            point_loss_weight=args.point_loss_weight,
            pair_loss_weight=args.pair_loss_weight,
            consistency_loss_weight=args.consistency_loss_weight,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            point_loss_weight=args.point_loss_weight,
            pair_loss_weight=args.pair_loss_weight,
            consistency_loss_weight=args.consistency_loss_weight,
        ) if len(test_pairs) > 0 else {
            "loss": 0.0,
            "point_mse": 0.0,
            "pair_bce": 0.0,
            "consistency_mse": 0.0,
            "label_acc": 0.0,
            "correct": 0,
            "num_pairs": 0,
        }

        history["train_loss"].append(train_metrics["loss"])
        history["train_point_mse"].append(train_metrics["point_mse"])
        history["train_pair_bce"].append(train_metrics["pair_bce"])
        history["train_consistency_mse"].append(train_metrics["consistency_mse"])
        history["train_label_acc"].append(train_metrics["label_acc"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_point_mse"].append(test_metrics["point_mse"])
        history["test_pair_bce"].append(test_metrics["pair_bce"])
        history["test_consistency_mse"].append(test_metrics["consistency_mse"])
        history["test_label_acc"].append(test_metrics["label_acc"])

        print(f"Epoch {epoch:03d}:")
        print(
            f"  Train - Label-ACC: {train_metrics['label_acc']:.4f} "
            f"({train_metrics['correct']}/{train_metrics['num_pairs']}), "
            f"Point-MSE: {train_metrics['point_mse']:.4f}, "
            f"Pair-BCE: {train_metrics['pair_bce']:.4f}, "
            f"Consistency: {train_metrics['consistency_mse']:.4f}, "
            f"Hybrid-Loss: {train_metrics['loss']:.4f}"
        )
        print(
            f"  Test  - Label-ACC: {test_metrics['label_acc']:.4f} "
            f"({test_metrics['correct']}/{test_metrics['num_pairs']}), "
            f"Point-MSE: {test_metrics['point_mse']:.4f}, "
            f"Pair-BCE: {test_metrics['pair_bce']:.4f}, "
            f"Consistency: {test_metrics['consistency_mse']:.4f}, "
            f"Hybrid-Loss: {test_metrics['loss']:.4f}"
        )

        if test_metrics["label_acc"] >= best_acc:
            best_acc = test_metrics["label_acc"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "args": vars(args),
                    "metric_mean": transform.mean,
                    "metric_std": transform.std,
                    "best_test_label_acc": best_acc,
                },
                best_ckpt,
            )

        save_json(history_path, history)

    print(f"Best model saved to: {best_ckpt}")
    print(f"History saved to: {history_path}")


if __name__ == "__main__":
    main(build_argparser().parse_args())
