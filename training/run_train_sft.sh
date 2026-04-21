#!/bin/bash
# SFT Training with Accelerate (Multi-GPU Support)

cd "$(dirname "$(readlink -f "$0")")/../.."

# GPU 配置
GPU_IDS="1"  # 使用空闲 GPU
NUM_GPUS=1

# 训练参数
SFT_DATA="data/pareto_true/pareto_and_near_pareto.jsonl"
EVAL_DATA="data/sft/designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"
BASE_MODEL="Qwen/Qwen2.5-Coder-7B-Instruct"
OUTPUT_DIR="./sft_output/pareto_near_pareto_no_reasoning"
BATCH_SIZE=4  # 每个 GPU 的 batch size
GRAD_ACCUM=4  # 有效 batch size = 4 * 4 = 16
EPOCHS=2
LEARNING_RATE=5e-6
MAX_LENGTH=2048
WEIGHT_DECAY=0.05
EVAL_STEPS=100
SAVE_STEPS=100

export CUDA_VISIBLE_DEVICES=$GPU_IDS

echo "=========================================="
echo "SFT Training with Accelerate"
echo "=========================================="
echo "GPUs: $GPU_IDS (Count: $NUM_GPUS)"
echo "Global batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Per-device batch size: $BATCH_SIZE"
echo "Base model: $BASE_MODEL"
echo "SFT data: $SFT_DATA"
echo "Eval data: $EVAL_DATA"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Check if SFT data exists
if [ ! -f "$SFT_DATA" ]; then
    echo "Error: SFT data not found: $SFT_DATA"
    echo ""
    echo "Please prepare SFT data first:"
    echo "  python scripts/split_train_data.py \\"
    echo "    --input data/designs_with_metrics_train.jsonl \\"
    echo "    --strategy sft_all_dpo_subset"
    exit 1
fi

if [ ! -f "$EVAL_DATA" ]; then
    echo "Error: eval data not found: $EVAL_DATA"
    exit 1
fi

echo "Starting SFT training with Accelerate..."
echo ""

# Use accelerate launch for multi-GPU training
# Using BF16 for better stability on H200 GPUs
accelerate launch \
    --num_processes=$NUM_GPUS \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m src.train_sft \
    --sft_data $SFT_DATA \
    --eval_data $EVAL_DATA \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --epochs $EPOCHS \
    --lr $LEARNING_RATE \
    --max_length $MAX_LENGTH \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --use_peft \
    --gradient_checkpointing

echo ""
echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
