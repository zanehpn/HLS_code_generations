"""
DPO (Direct Preference Optimization) for PPA-HLS.

Second stage of SFT-RL: use preference data (chosen > rejected) to optimize for PPA.
Uses Accelerate for efficient multi-GPU training.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback, EarlyStoppingCallback
from trl import DPOTrainer, DPOConfig
from datasets import Dataset
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from .acceleration_utils import build_deepspeed_config, ensure_optional_dependency
from .dataset import load_dpo_jsonl, load_dpo_from_designs_jsonl, DPO_SYSTEM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DPO_EVAL_DATA = PROJECT_ROOT / "data" / "dpo" / "dpo_test_latency_top_bottom_app_disjoint_test_108.jsonl"
LEGACY_DEFAULT_EVAL_DATA = (
    PROJECT_ROOT / "data" / "sft" / "designs_with_metrics_app_disjoint_test_108_apps.jsonl"
)
DEFAULT_EVAL_DATA = str(DEFAULT_DPO_EVAL_DATA if DEFAULT_DPO_EVAL_DATA.exists() else LEGACY_DEFAULT_EVAL_DATA)


class LossLoggingCallback(TrainerCallback):
    """Save train and eval loss to JSONL file."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def _append_log(self, record: dict, is_main: bool):
        if is_main:
            with open(self.output_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs and state.is_world_process_zero:
            record = {
                "phase": "train",
                "step": state.global_step,
                "epoch": state.epoch,
                "train_loss": round(logs["loss"], 6),
                "eval_loss": None,
            }
            self._append_log(record, state.is_world_process_zero)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics and state.is_world_process_zero:
            record = {
                "phase": "eval",
                "step": state.global_step,
                "epoch": state.epoch,
                "train_loss": None,
                "eval_loss": round(metrics["eval_loss"], 6),
            }
            self._append_log(record, state.is_world_process_zero)


def _format_prompt(prompt: str) -> str:
    """Format user prompt with system message."""
    return (
        f"<|im_start|>system\n{DPO_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpo_data", type=str, required=True, help="Path to DPO preference JSONL")
    parser.add_argument("--base_model", type=str, required=True, help="Base or SFT checkpoint")
    parser.add_argument("--output_dir", type=str, default="./dpo_output")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device batch size (increased from 1 for better gradient estimate)")
    parser.add_argument("--lr", type=float, default=5e-7, help="Learning rate (increased from 2e-7 to improve learning)")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta (lowered from 0.2 to reduce regularization)")
    parser.add_argument("--max_length", type=int, default=2048, help="Max sequence length (reduce if OOM)")
    parser.add_argument("--max_prompt_length", type=int, default=512, help="Max prompt length (reduce if OOM)")
    parser.add_argument("--gpu", type=str, default=None, help="GPU device(s) to use (e.g., '2' or '0,1')")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True, help="Enable gradient checkpointing")
    parser.add_argument("--save_steps", type=int, default=250, help="Save checkpoint every N steps (aligned with eval_steps)")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Keep at most N checkpoints")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume from latest checkpoint if exists")
    parser.add_argument("--precompute_ref_log_probs", action="store_true", default=True,
                        help="Precompute ref log probs to save ~14GB (no ref model in memory during training)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Gradient accumulation (increased from 4 for more stable gradients)")
    parser.add_argument("--use_deepspeed", action="store_true", help="Enable DeepSpeed ZeRO optimization")
    parser.add_argument("--deepspeed_zero_stage", type=int, choices=[0, 1, 2, 3], default=2,
                        help="DeepSpeed ZeRO stage")
    parser.add_argument("--deepspeed_offload_optimizer_device", type=str, choices=["none", "cpu", "nvme"],
                        default="none", help="DeepSpeed optimizer offload target")
    parser.add_argument("--deepspeed_offload_param_device", type=str, choices=["none", "cpu", "nvme"],
                        default="none", help="DeepSpeed parameter offload target (ZeRO-3 only)")
    parser.add_argument("--eval_data", type=str, default=DEFAULT_EVAL_DATA,
                        help="Path to eval data (DPO JSONL or designs JSONL); empty string disables eval")
    parser.add_argument("--eval_steps", type=int, default=250, help="Evaluate every N steps (more frequent for better monitoring)")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                        help="Stop if eval_loss doesn't improve for N evals (increased from 3 for more patience)")
    args = parser.parse_args()
    deepspeed_config = None
    if args.use_deepspeed:
        ensure_optional_dependency("deepspeed", "train_dpo.py --use_deepspeed", pip_name="deepspeed")
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
    # Note: H200 supports BF16 which is more stable than FP16 for training
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator = Accelerator(
        mixed_precision="bf16",  # Use BF16 instead of FP16 for better stability
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs],
    )
    
    # Check available GPUs
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print("Accelerate Configuration")
        print(f"{'='*60}")
        print(f"Distributed type: {accelerator.distributed_type}")
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Process index: {accelerator.process_index}")
        print(f"Local process index: {accelerator.local_process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"{'='*60}\n")

    if accelerator.is_main_process:
        print("Loading tokenizer and model...")
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    # Load models with appropriate dtype (use bfloat16 for better stability on H200)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if accelerator.is_main_process:
            print("Gradient checkpointing enabled")

    # Only load ref_model if not precomputing (saves ~14GB when precompute_ref_log_probs=True)
    ref_model = None
    if not args.precompute_ref_log_probs:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
    elif accelerator.is_main_process:
        print("Precomputing ref log probs (ref_model not loaded, saves ~14GB VRAM)")

    if accelerator.is_main_process:
        print(f"\nLoading training data from: {args.dpo_data}")
    
    data = load_dpo_jsonl(args.dpo_data)
    if not data:
        raise ValueError(f"No DPO data found in {args.dpo_data}")
    
    if accelerator.is_main_process:
        print(f"Loaded {len(data)} preference pairs")

    # TRL DPO: prompt = prefix before assistant response, chosen/rejected = assistant response only
    dataset_dict = {
        "prompt": [_format_prompt(d["prompt"]) for d in data],
        "chosen": [d["chosen"] + "<|im_end|>" for d in data],
        "rejected": [d["rejected"] + "<|im_end|>" for d in data],
    }
    train_dataset = Dataset.from_dict(dataset_dict)

    eval_requested = bool(args.eval_data.strip()) and args.eval_steps > 0

    # Load eval dataset (DPO JSONL or designs JSONL)
    eval_dataset = None
    if not eval_requested:
        if accelerator.is_main_process:
            print("Eval disabled: pass a non-empty --eval_data and --eval_steps > 0 to enable best-checkpoint selection.")
    elif os.path.exists(args.eval_data):
        eval_data = load_dpo_jsonl(args.eval_data)
        eval_source = "dpo_jsonl"
        if not eval_data:
            eval_data = load_dpo_from_designs_jsonl(args.eval_data)
            eval_source = "designs_jsonl"
        if eval_data and accelerator.is_main_process:
            print(f"Loading eval data from: {args.eval_data}")
            print(f"Loaded {len(eval_data)} eval preference pairs from {eval_source}")
        if eval_data:
            eval_dict = {
                "prompt": [_format_prompt(d["prompt"]) for d in eval_data],
                "chosen": [d["chosen"] + "<|im_end|>" for d in eval_data],
                "rejected": [d["rejected"] + "<|im_end|>" for d in eval_data],
            }
            eval_dataset = Dataset.from_dict(eval_dict)
        elif accelerator.is_main_process and args.eval_steps > 0:
            print(
                f"Eval data exists but produced 0 preference pairs: {args.eval_data}\n"
                "DPO eval needs either prompt/chosen/rejected JSONL or a designs JSONL with at least 2 designs per algorithm."
            )
    elif accelerator.is_main_process:
        print(f"Eval data not found: {args.eval_data}, skipping evaluation during training")

    use_early_stopping = eval_dataset is not None and args.early_stopping_patience > 0
    save_steps = args.save_steps
    if use_early_stopping and args.save_steps % args.eval_steps != 0:
        save_steps = args.eval_steps  # align save with eval for load_best_model_at_end

    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        beta=args.beta,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        load_best_model_at_end=use_early_stopping,
        metric_for_best_model="eval_loss" if use_early_stopping else None,
        greater_is_better=False if use_early_stopping else None,
        bf16=True,  # Use BF16 instead of FP16 (more stable on H200)
        fp16=False,  # Disable FP16 to avoid conflicts
        remove_unused_columns=False,
        report_to="none",
        # Memory optimization: precompute ref log probs saves ~14GB (no ref model in memory)
        precompute_ref_log_probs=args.precompute_ref_log_probs,
        precompute_ref_batch_size=8,
        # Accelerate/DDP configuration
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        deepspeed=deepspeed_config,
    )

    if accelerator.is_main_process:
        print("\nInitializing DPO Trainer with Accelerate...")
    
    # Note: DPOTrainer in newer versions of TRL uses processing_class instead of tokenizer
    # or gets tokenizer from DPOConfig
    loss_log_path = os.path.join(args.output_dir, "train_eval_losses.jsonl")
    callbacks = [LossLoggingCallback(loss_log_path)]
    if use_early_stopping:
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=args.early_stopping_patience,
            early_stopping_threshold=0.0,
        ))

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Check for existing checkpoint to resume
    checkpoint_dir = None
    if args.resume_from_checkpoint and accelerator.is_main_process and os.path.exists(args.output_dir):
        checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            latest = max(checkpoints, key=lambda x: int(x.split("-")[1]))
            checkpoint_dir = os.path.join(args.output_dir, latest)
            print(f"\nResuming from checkpoint: {checkpoint_dir}\n")

    if accelerator.is_main_process:
        print("\n" + "="*60)
        print("Starting DPO Training")
        print("="*60)
        print(f"Training samples: {len(train_dataset)}")
        print(f"Batch size per device: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"Epochs: {args.epochs}")
        print(f"Learning rate: {args.lr}")
        print(f"Beta: {args.beta}")
        if args.use_deepspeed:
            print(
                f"DeepSpeed enabled: ZeRO-{args.deepspeed_zero_stage}, "
                f"optimizer_offload={args.deepspeed_offload_optimizer_device}, "
                f"param_offload={args.deepspeed_offload_param_device}"
            )
        print(f"Save every {save_steps} steps, keep {args.save_total_limit} checkpoints")
        print(f"Precompute ref log probs: {args.precompute_ref_log_probs}")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
        if eval_dataset:
            print(f"Eval every {args.eval_steps} steps ({len(eval_dataset)} preference pairs), losses saved to {loss_log_path}")
        elif args.eval_steps > 0:
            print("Eval disabled: no valid eval dataset was loaded, so train_eval_losses.jsonl will contain only train-phase records.")
        if use_early_stopping:
            print(f"Early stopping: patience={args.early_stopping_patience} evals, load_best_model_at_end=True")
            print("Best checkpoint selection: enabled (metric=eval_loss, lower is better)")
        print("="*60 + "\n")

    trainer.train(resume_from_checkpoint=checkpoint_dir)
    
    # Save model only on main process
    if accelerator.is_main_process:
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nDPO model saved to {args.output_dir}")
        print("Training completed successfully!")


if __name__ == "__main__":
    main()
