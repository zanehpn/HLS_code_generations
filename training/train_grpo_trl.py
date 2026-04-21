"""
GRPO training for NL2HLS using TRL + vLLM.

Drop-in replacement for train_grpo.py with:
- vLLM for fast generation
- DeepSpeed Zero3 for multi-GPU
- TRL's GRPOTrainer for cleaner training loop
- Same reward functions: compile, sim, QoR rerank
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_callback import TrainerCallback

from trl import GRPOConfig, GRPOTrainer

# Import reward utilities from existing train_grpo
from .train_grpo import (
    PROJECT_ROOT,
    TB_COMPAT_DIR,
    _extract_code_from_generation,
    _parse_compile_and_sim_scores,
    _parse_test_case_result,
    check_reasoning_format,
    initialize_reward_backend,
    load_prompt_jsonl,
    run_group_rerank,
    test_case,
    TEXT_REWARD_RUNTIME,
    REWARD_BACKEND,
)


# ---------------------
# Reward Functions
# ---------------------

def compile_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward: 1.0 if code compiles, 0.0 otherwise."""
    metadata_list = kwargs.get("metadata", [{}] * len(prompts))
    if not metadata_list or not isinstance(metadata_list[0], dict):
        metadata_list = [{}] * len(prompts)
    rewards = []
    for prompt, completion, meta in zip(prompts, completions, metadata_list):
        tc_out = test_case(prompt, completion, meta)
        _, compile_score, _ = _parse_compile_and_sim_scores(tc_out)
        rewards.append(float(compile_score))
    return rewards


def sim_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward: 1.0 if functional simulation passes, 0.0 otherwise."""
    metadata_list = kwargs.get("metadata", [{}] * len(prompts))
    rewards = []
    for prompt, completion, meta in zip(prompts, completions, metadata_list):
        tc_out = test_case(prompt, completion, meta)
        _, _, sim_score = _parse_compile_and_sim_scores(tc_out)
        rewards.append(float(sim_score))
    return rewards


def qor_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward: QoR score from pairwise text ranker (group rerank)."""
    metadata_list = kwargs.get("metadata", [{}] * len(prompts))
    rewards = []

    # Group by prompt for pairwise reranking
    # trl calls reward_fn with all completions for all prompts in a flat list
    # We need to figure out the group structure
    num_prompts = len(set(prompts))
    group_size = len(prompts) // num_prompts if num_prompts > 0 else 1

    for start in range(0, len(prompts), group_size):
        group_prompts = prompts[start:start + group_size]
        group_completions = completions[start:start + group_size]
        group_meta = metadata_list[start:start + group_size]

        # Get testcase results for gating
        tc_results = []
        for prompt, comp, meta in zip(group_prompts, group_completions, group_meta):
            tc_results.append(test_case(prompt, comp, meta))

        group_rerank = run_group_rerank(
            prompt=group_prompts[0],
            generated_codes=group_completions,
            metadata_list=group_meta,
            test_case_results=tc_results,
        )

        for out in group_rerank:
            score = float(out.get("score", 0.0)) if isinstance(out, dict) else 0.0
            rewards.append(score)

    return rewards


def format_reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward: 1.0 if output has <think>...</think><final_code>...</final_code> format."""
    return [1.0 if check_reasoning_format(c) else 0.0 for c in completions]


# ---------------------
# Dataset
# ---------------------

def build_dataset(data_path: str, max_samples: int = 0) -> Dataset:
    """Load JSONL and convert to HuggingFace Dataset for trl."""
    samples = load_prompt_jsonl(data_path)
    if max_samples > 0:
        samples = samples[:max_samples]

    records = []
    for s in samples:
        prompt_text = (
            f"<|im_start|>user\n{s['prompt']}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        records.append({
            "prompt": prompt_text,
            "metadata": s.get("metadata", {}),
        })

    return Dataset.from_list(records)


# ---------------------
# Callbacks
# ---------------------

class BestCheckpointCallback(TrainerCallback):
    """Track best eval reward, save best checkpoint, write train_log.jsonl."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.best_eval_reward = -float("inf")
        self.best_step = -1
        self.log_path = Path(output_dir) / "train_log.jsonl"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        entry = {"type": "train", "step": step}
        for k, v in logs.items():
            if isinstance(v, (int, float)):
                entry[k] = round(v, 6) if isinstance(v, float) else v
        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()

        # trl logs eval metrics via on_log (not on_evaluate), check for eval_reward
        eval_reward = logs.get("eval_reward")
        if eval_reward is not None and eval_reward > self.best_eval_reward:
            eval_sim = logs.get("eval_rewards/sim_reward_fn/mean", 0)
            eval_compile = logs.get("eval_rewards/compile_reward_fn/mean", 0)
            self.best_eval_reward = eval_reward
            self.best_step = step
            best_dir = os.path.join(self.output_dir, "best_checkpoint")
            model = kwargs.get("model")
            tokenizer = kwargs.get("processing_class") or kwargs.get("tokenizer")
            if model is not None:
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                if tokenizer is not None:
                    tokenizer.save_pretrained(best_dir)
                with open(os.path.join(best_dir, "eval_summary.json"), "w") as f:
                    json.dump({
                        "step": step,
                        "eval_reward": round(self.best_eval_reward, 4),
                        "eval_compile": round(eval_compile, 4),
                        "eval_sim": round(eval_sim, 4),
                    }, f)
                print(f"[step {step}] New best eval reward={self.best_eval_reward:.4f} compile={eval_compile:.4f} sim={eval_sim:.4f} (saved to {best_dir})")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        step = state.global_step
        entry = {"type": "eval", "step": step}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                entry[k] = round(v, 6) if isinstance(v, float) else v
        self.log_file.write(json.dumps(entry) + "\n")
        self.log_file.flush()

    def on_train_end(self, args, state, control, **kwargs):
        self.log_file.close()
        summary = {
            "best_step": self.best_step,
            "best_eval_reward": round(self.best_eval_reward, 4) if self.best_eval_reward > -float("inf") else None,
        }
        with open(os.path.join(self.output_dir, "best_eval_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nTraining done. Best eval reward={self.best_eval_reward:.4f} at step {self.best_step}")


# ---------------------
# Main
# ---------------------

def main():
    parser = argparse.ArgumentParser(description="GRPO training with TRL + vLLM")
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--eval_data", type=str, default="")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./grpo_output/trl_grpo")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--kl_coef", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=0)

    # Reward weights: compile, sim, qor, format
    parser.add_argument("--reward_compile_weight", type=float, default=0.3)
    parser.add_argument("--reward_sim_weight", type=float, default=0.3)
    parser.add_argument("--reward_qor_weight", type=float, default=0.3)
    parser.add_argument("--reward_format_weight", type=float, default=0.0)
    parser.add_argument("--reasoning_mode", action="store_true", default=False)

    # vLLM
    parser.add_argument("--use_vllm", action="store_true", default=True)
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.7)

    # LoRA
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Text ranker
    parser.add_argument("--text_ranker_ckpt", type=str,
                        default=str(PROJECT_ROOT / "GNN" / "models_text_ranker_unixcoder" / "best_pair_text_ranker.pt"))
    parser.add_argument("--text_ranker_ref_jsonl", type=str,
                        default=f"{PROJECT_ROOT / 'data' / 'sft' / 'designs_with_metrics_train.jsonl'},{PROJECT_ROOT / 'data' / 'designs_with_metrics_test.jsonl'}")
    parser.add_argument("--text_ranker_model_name", type=str, default="microsoft/unixcoder-base")
    parser.add_argument("--text_ranker_max_length", type=int, default=256)

    # Logging
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=str, default=None)

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # Build reward functions and weights
    reward_funcs = [compile_reward_fn, sim_reward_fn, qor_reward_fn]
    reward_weights = [args.reward_compile_weight, args.reward_sim_weight, args.reward_qor_weight]

    if args.reasoning_mode and args.reward_format_weight > 0:
        reward_funcs.append(format_reward_fn)
        reward_weights.append(args.reward_format_weight)

    # Build dataset
    train_dataset = build_dataset(args.train_data, args.max_samples)
    eval_dataset = None
    if args.eval_data and os.path.exists(args.eval_data):
        eval_dataset = build_dataset(args.eval_data)

    print(f"Train dataset: {len(train_dataset)} samples")
    if eval_dataset:
        print(f"Eval dataset: {len(eval_dataset)} samples")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.num_generations,  # must be divisible by num_generations
        learning_rate=args.lr,
        num_generations=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        max_completion_length=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        beta=args.kl_coef,
        reward_weights=reward_weights,
        use_vllm=args.use_vllm,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        seed=args.seed,
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
    )

    # Initialize reward backend (text ranker)
    class FakeAccelerator:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class FakeArgs:
        pass

    fake_args = FakeArgs()
    fake_args.reward_backend = "text_ranker"
    fake_args.text_ranker_ckpt = args.text_ranker_ckpt
    fake_args.text_ranker_ref_jsonl = args.text_ranker_ref_jsonl
    fake_args.text_ranker_model_name = args.text_ranker_model_name
    fake_args.text_ranker_max_length = args.text_ranker_max_length
    fake_args.mc_dropout_passes = 10
    fake_args.uncertainty_threshold = 0.1
    initialize_reward_backend(fake_args, FakeAccelerator())

    # Create trainer with callbacks
    best_ckpt_callback = BestCheckpointCallback(output_dir=args.output_dir)

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=reward_funcs,
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=[best_ckpt_callback],
    )

    print("=" * 50)
    print("GRPO Training with TRL + vLLM")
    print(f"Base model: {args.base_model}")
    print(f"Reward weights: compile={args.reward_compile_weight} sim={args.reward_sim_weight} qor={args.reward_qor_weight} fmt={args.reward_format_weight}")
    print(f"vLLM: {args.use_vllm} (gpu_mem={args.vllm_gpu_memory_utilization})")
    print(f"KL coef: {args.kl_coef}")
    print("=" * 50)

    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"\nTraining done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
