"""
SFT (Supervised Fine-Tuning) for NL2HLS base model.

First stage of SFT-RL: train the model to generate HLS code from functional descriptions.
Uses Accelerate for efficient multi-GPU training.
"""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from .acceleration_utils import build_deepspeed_config, ensure_optional_dependency
from .dataset import load_sft_jsonl, sft_data_to_hf_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_DATA = str(
    PROJECT_ROOT / "data" / "sft" / "designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"
)


class LossLoggingCallback(TrainerCallback):
    """Save train and eval loss to JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def _append_log(self, record: dict, is_main: bool):
        if is_main:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and state.is_world_process_zero:
            record = {
                "step": state.global_step,
                "epoch": state.epoch,
                "train_loss": round(logs["loss"], 6),
                "eval_loss": None,
            }
            self._append_log(record, state.is_world_process_zero)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics and state.is_world_process_zero:
            record = {
                "step": state.global_step,
                "epoch": state.epoch,
                "train_loss": None,
                "eval_loss": round(metrics["eval_loss"], 6),
            }
            self._append_log(record, state.is_world_process_zero)


def _build_resource_limits(args: argparse.Namespace) -> dict:
    resource_limits = {}
    for metric_name, arg_name in [
        ("LUT", "max_lut"),
        ("FF", "max_ff"),
        ("DSP", "max_dsp"),
        ("BRAM_18K", "max_bram_18k"),
        ("TargetClockPeriod", "max_target_clock_period"),
        ("Worst-caseLatency", "max_worst_case_latency"),
        ("Best-caseLatency", "max_best_case_latency"),
    ]:
        metric_limit = getattr(args, arg_name)
        if metric_limit is not None:
            resource_limits[metric_name] = float(metric_limit)
    return resource_limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft_data", type=str, required=True, help="Path to SFT JSONL")
    parser.add_argument("--eval_data", type=str, default=DEFAULT_EVAL_DATA,
                        help=f"Path to eval JSONL (default: {DEFAULT_EVAL_DATA})")
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (reduced to mitigate overfitting)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for regularization")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--tokenize_batch_size", type=int, default=256,
                        help="Batch size for tokenization map (lower if host RAM is limited)")
    parser.add_argument("--use_peft", dest="use_peft", action="store_true", help="Enable LoRA PEFT")
    parser.add_argument("--no_peft", dest="use_peft", action="store_false", help="Disable LoRA PEFT")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout for regularization")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", dest="gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false",
                        help="Disable gradient checkpointing")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed ZeRO optimization")
    parser.add_argument("--deepspeed_zero_stage", type=int, choices=[0, 1, 2, 3], default=2,
                        help="DeepSpeed ZeRO stage")
    parser.add_argument("--deepspeed_offload_optimizer_device", type=str, choices=["none", "cpu", "nvme"],
                        default="none", help="DeepSpeed optimizer offload target")
    parser.add_argument("--deepspeed_offload_param_device", type=str, choices=["none", "cpu", "nvme"],
                        default="none", help="DeepSpeed parameter offload target (ZeRO-3 only)")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device(s) to use (e.g., '2' or '0,1')")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--eval_samples", type=int, default=100,
                        help="Random sample N examples for eval (0 = use full eval set, default 100 to speed up)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Keep at most N checkpoints")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint if exists")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop if eval_loss doesn't improve for N evals (0=disabled, default=5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for eval subset sampling")
    parser.add_argument("--max_lut", type=float, default=None, help="Drop SFT examples with LUT above this platform budget")
    parser.add_argument("--max_ff", type=float, default=None, help="Drop SFT examples with FF above this platform budget")
    parser.add_argument("--max_dsp", type=float, default=None, help="Drop SFT examples with DSP above this platform budget")
    parser.add_argument("--max_bram_18k", type=float, default=None, help="Drop SFT examples with BRAM_18K above this platform budget")
    parser.add_argument("--max_target_clock_period", type=float, default=None, help="Drop SFT examples with TargetClockPeriod above this platform budget")
    parser.add_argument("--max_worst_case_latency", type=float, default=None, help="Drop SFT examples with Worst-caseLatency above this platform budget")
    parser.add_argument("--max_best_case_latency", type=float, default=None, help="Drop SFT examples with Best-caseLatency above this platform budget")
    parser.set_defaults(use_peft=True, gradient_checkpointing=True)
    args = parser.parse_args()
    resource_limits = _build_resource_limits(args)
    deepspeed_config = None
    if args.use_deepspeed:
        ensure_optional_dependency("deepspeed", "train_sft.py --use_deepspeed", pip_name="deepspeed")
        deepspeed_config = build_deepspeed_config(
            train_micro_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            zero_stage=args.deepspeed_zero_stage,
            bf16=torch.cuda.is_available(),
            gradient_clipping=1.0,
            offload_optimizer_device=args.deepspeed_offload_optimizer_device,
            offload_param_device=args.deepspeed_offload_param_device,
        )

    # Set GPU device
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU(s): {args.gpu}")

    # Initialize Accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("Accelerate Configuration")
        print(f"{'='*60}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Device: {accelerator.device}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

    if accelerator.is_main_process:
        print(f"Loading tokenizer and model from: {args.base_model}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load base model '{args.base_model}': {e}") from e

    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_config)
        if accelerator.is_main_process:
            model.print_trainable_parameters()

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")

    if accelerator.is_main_process:
        print(f"\nLoading training data from: {args.sft_data}")
        if resource_limits:
            print(f"Applying platform resource limits to SFT data: {resource_limits}")
    data = load_sft_jsonl(args.sft_data, resource_limits=resource_limits)
    if not data:
        raise ValueError(f"No SFT data found in {args.sft_data} after applying resource limits {resource_limits}")
    if accelerator.is_main_process:
        print(f"Loaded {len(data)} training examples")

    train_dataset = sft_data_to_hf_dataset(
        data=data,
        tokenizer=tokenizer,
        max_length=args.max_length,
        map_batch_size=args.tokenize_batch_size,
    )

    # Load eval dataset
    eval_dataset = None
    if os.path.exists(args.eval_data):
        eval_data = load_sft_jsonl(args.eval_data, resource_limits=resource_limits)
        if eval_data and accelerator.is_main_process:
            print(f"Loading eval data from: {args.eval_data}")
            print(f"Loaded {len(eval_data)} eval examples")
        if eval_data:
            if args.eval_samples > 0 and len(eval_data) > args.eval_samples:
                rng = random.Random(args.seed)
                indices = sorted(rng.sample(range(len(eval_data)), args.eval_samples))
                eval_data = [eval_data[i] for i in indices]
                if accelerator.is_main_process:
                    print(f"Eval: randomly sampled {args.eval_samples} examples (seed={args.seed})")
            eval_dataset = sft_data_to_hf_dataset(
                data=eval_data,
                tokenizer=tokenizer,
                max_length=args.max_length,
                map_batch_size=args.tokenize_batch_size,
            )
    elif accelerator.is_main_process:
        print(f"Eval data not found: {args.eval_data}, skipping evaluation during training")

    use_early_stopping = eval_dataset is not None and args.early_stopping_patience > 0
    save_steps = args.save_steps
    if use_early_stopping and args.save_steps % args.eval_steps != 0:
        save_steps = args.eval_steps  # align save with eval for load_best_model_at_end

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        bf16=True,
        fp16=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to="none",
        ddp_find_unused_parameters=False,
        logging_dir=f"{args.output_dir}/logs",
        logging_first_step=True,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed=deepspeed_config,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    loss_log_path = os.path.join(args.output_dir, "train_eval_losses.jsonl")
    callbacks = [LossLoggingCallback(loss_log_path)]
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.0,
        ))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Check for existing checkpoint to resume
    checkpoint_dir = None
    if args.resume_from_checkpoint and os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(args.output_dir, latest)
            if accelerator.is_main_process:
                print(f"\nResuming from checkpoint: {checkpoint_dir}\n")

    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("Starting SFT Training")
        print(f"{'='*60}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Batch size per device: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        if args.use_deepspeed:
            print(
                f"DeepSpeed enabled: ZeRO-{args.deepspeed_zero_stage}, "
                f"optimizer_offload={args.deepspeed_offload_optimizer_device}, "
                f"param_offload={args.deepspeed_offload_param_device}"
            )
        print(f"Save every {save_steps} steps, keep {args.save_total_limit} checkpoints")
        if eval_dataset:
            print(f"Eval every {args.eval_steps} steps ({len(eval_dataset)} samples), losses saved to {loss_log_path}")
        if use_early_stopping:
            print(f"Early stopping: patience={args.early_stopping_patience} evals, load_best_model_at_end=True")
        print(f"LoRA dropout: {args.lora_dropout}")
        print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=checkpoint_dir)

    if accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n{'='*50}")
        print(f"SFT model saved to {args.output_dir}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
