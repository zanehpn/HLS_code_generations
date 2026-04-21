#!/bin/bash
# Complete SFT + DPO Training Pipeline

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
GPU_IDS=${1:-2,3}
STRATEGY=${2:-sft_all_dpo_subset}
DPO_RATIO=${3:-0.5}
SFT_TRAIN_DATA="data/sft/designs_with_metrics_app_disjoint_train_rest_after_test_108_apps.jsonl"
SFT_EVAL_DATA="data/sft/designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"
DPO_SPLIT_INPUT="$SFT_TRAIN_DATA"
UNUSED_SPLIT_SFT_DATA="data/_unused_split_train_sft.jsonl"
DPO_SOURCE_DATA="data/train_dpo_source.jsonl"
DPO_PAIRS_DATA="data/dpo_pairs_top_bottom.jsonl"
DEFAULT_DPO_EVAL_SOURCE_DATA="data/sft/designs_with_metrics_app_disjoint_test_108_apps.jsonl"
DEFAULT_DPO_EVAL_DATA="data/dpo/dpo_test_latency_top_bottom_app_disjoint_test_108.jsonl"
SFT_MODEL_DIR="output/sft_pre_dpo"
DPO_MODEL_DIR="output/dpo_model"

RAW_DPO_EVAL_DATA=${4:-$DEFAULT_DPO_EVAL_DATA}
case "${RAW_DPO_EVAL_DATA,,}" in
    ""|none|disable|disabled)
        DPO_EVAL_DATA=""
        ;;
    *)
        DPO_EVAL_DATA="$RAW_DPO_EVAL_DATA"
        ;;
esac

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"

echo "=========================================="
echo "NL2HLS Complete Training Pipeline"
echo "=========================================="
echo "GPUs: $GPU_IDS (Count: $NUM_GPUS)"
echo "Split Strategy: $STRATEGY"
echo "DPO Ratio: $DPO_RATIO"
if [ -n "$DPO_EVAL_DATA" ]; then
    echo "DPO eval data: $DPO_EVAL_DATA"
else
    echo "DPO eval data: <disabled>"
fi
echo ""

cd "$PROJECT_ROOT"

# ============================================
# Step 1: Prepare DPO Source Data
# ============================================
echo "=========================================="
echo "Step 1: Preparing DPO source data..."
echo "=========================================="

if [ ! -f "$SFT_TRAIN_DATA" ]; then
    echo "Error: SFT train data not found: $SFT_TRAIN_DATA"
    exit 1
fi

if [ ! -f "$SFT_EVAL_DATA" ]; then
    echo "Error: SFT eval/test data not found: $SFT_EVAL_DATA"
    exit 1
fi

if [ ! -f "$DPO_SPLIT_INPUT" ]; then
    echo "Error: DPO split input not found: $DPO_SPLIT_INPUT"
    exit 1
fi

python scripts/split_train_data.py \
    --input "$DPO_SPLIT_INPUT" \
    --strategy "$STRATEGY" \
    --dpo_ratio "$DPO_RATIO" \
    --sft_output "$UNUSED_SPLIT_SFT_DATA" \
    --dpo_output "$DPO_SOURCE_DATA" \
    --seed 42

if [ $? -ne 0 ]; then
    echo "Error: Data split failed"
    exit 1
fi

echo ""
echo "DPO source data ready!"
echo "  SFT train data: $SFT_TRAIN_DATA"
echo "  SFT eval/test data: $SFT_EVAL_DATA"
echo "  DPO split input: $DPO_SPLIT_INPUT"
echo "  DPO source data: $DPO_SOURCE_DATA"
echo ""
read -p "Press Enter to continue to SFT training..."

# ============================================
# Step 2: SFT Training
# ============================================
echo ""
echo "=========================================="
echo "Step 2: Training SFT model..."
echo "=========================================="
echo "Launching SFT with accelerate on GPUs: $GPU_IDS"
accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m src.train_sft \
    --sft_data "$SFT_TRAIN_DATA" \
    --eval_data "$SFT_EVAL_DATA" \
    --output_dir "$SFT_MODEL_DIR" \
    --base_model Qwen/Qwen2.5-Coder-7B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --lr 2e-5

if [ $? -ne 0 ]; then
    echo "Error: SFT training failed"
    exit 1
fi

echo ""
echo "SFT training complete!"
echo "  Model saved to: $SFT_MODEL_DIR"
echo ""
read -p "Press Enter to continue to DPO data generation..."

# ============================================
# Step 3: Generate DPO Preference Pairs
# ============================================
echo ""
echo "=========================================="
echo "Step 3: Generating DPO preference pairs..."
echo "=========================================="
echo "Using top_vs_bottom strategy for better performance"
python scripts/generate_dpo_from_designs.py \
    --input "$DPO_SOURCE_DATA" \
    --output "$DPO_PAIRS_DATA" \
    --strategy top_vs_bottom \
    --metric Worst-caseLatency

if [ $? -ne 0 ]; then
    echo "Error: DPO pair generation failed"
    exit 1
fi

echo ""
echo "DPO pairs generated!"
echo "  Output: $DPO_PAIRS_DATA"
echo ""
read -p "Press Enter to continue to DPO training..."

# ============================================
# Step 4: DPO Training
# ============================================
echo ""
echo "=========================================="
echo "Step 4: Training DPO model..."
echo "=========================================="
if [ -n "$DPO_EVAL_DATA" ]; then
    if [ ! -f "$DPO_EVAL_DATA" ]; then
        if [ "$DPO_EVAL_DATA" = "$DEFAULT_DPO_EVAL_DATA" ] && [ -f "$DEFAULT_DPO_EVAL_SOURCE_DATA" ]; then
            echo "Default DPO eval pair file not found. Generating it from: $DEFAULT_DPO_EVAL_SOURCE_DATA"
            python scripts/generate_dpo_from_designs.py \
                --input "$DEFAULT_DPO_EVAL_SOURCE_DATA" \
                --output "$DPO_EVAL_DATA" \
                --strategy top_vs_bottom \
                --metric Worst-caseLatency
        else
            echo "Error: DPO eval data not found: $DPO_EVAL_DATA"
            exit 1
        fi
    fi
    if [ ! -f "$DPO_EVAL_DATA" ]; then
        echo "Error: Failed to prepare DPO eval data: $DPO_EVAL_DATA"
        exit 1
    fi
    echo "DPO eval enabled. train_dpo.py will select the best checkpoint by eval_loss, matching the SFT training behavior."
    DPO_EVAL_ARGS=(--eval_data "$DPO_EVAL_DATA")
else
    echo "DPO eval disabled. Pass a DPO eval JSONL as the 4th argument, or omit the 4th argument to use the default DPO eval set."
    DPO_EVAL_ARGS=(--eval_steps 0 --eval_data "")
fi
echo "Launching DPO with accelerate on GPUs: $GPU_IDS"
accelerate launch \
    --num_processes="$NUM_GPUS" \
    --num_machines=1 \
    --mixed_precision=bf16 \
    -m src.train_dpo \
    --dpo_data "$DPO_PAIRS_DATA" \
    --base_model "$SFT_MODEL_DIR" \
    --output_dir "$DPO_MODEL_DIR" \
    --epochs 1 \
    --batch_size 2 \
    --lr 5e-7 \
    "${DPO_EVAL_ARGS[@]}"

if [ $? -ne 0 ]; then
    echo "Error: DPO training failed"
    exit 1
fi

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "🎉 Training Pipeline Complete!"
echo "=========================================="
echo ""
echo "Models:"
echo "  SFT Model: $SFT_MODEL_DIR"
echo "  DPO Model: $DPO_MODEL_DIR"
echo ""
echo "Data Files:"
echo "  SFT train data: $SFT_TRAIN_DATA"
echo "  SFT eval/test data: $SFT_EVAL_DATA"
echo "  DPO split input: $DPO_SPLIT_INPUT"
echo "  DPO source: $DPO_SOURCE_DATA"
echo "  DPO pairs: $DPO_PAIRS_DATA"
if [ -n "$DPO_EVAL_DATA" ]; then
    echo "  DPO eval data: $DPO_EVAL_DATA"
else
    echo "  DPO eval data: <disabled>"
fi
echo ""
echo "Next Steps:"
echo "1. Test the models:"
echo "   python -m src.inference --model $DPO_MODEL_DIR --prompt 'your prompt'"
echo ""
echo "2. Compare SFT vs DPO:"
echo "   python -m src.inference --model $SFT_MODEL_DIR --prompt 'your prompt'"
echo "   python -m src.inference --model $DPO_MODEL_DIR --prompt 'your prompt'"
echo ""
echo "=========================================="
