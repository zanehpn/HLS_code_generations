#!/bin/bash
# GRPO Training with Reasoning Mode + Uncertainty-Aware Proxy Reward Switching
#
# Features:
#   1. Reasoning mode: <think>...</think> + <final_code>...</final_code>
#   2. Format reward: +1 for correct reasoning format
#   3. MC Dropout uncertainty estimation on reward model (M=10 passes)
#   4. When uncertainty > tau_u, fall back to real synthesis for ground-truth QoR
#   5. Online fine-tuning of reward model on replay buffer every K steps

cd "$(dirname "$(readlink -f "$0")")/../.."

# GPU 配置
GPU_IDS="0,1"

# 路径
BASE_MODEL="./sft_output/pareto_near_pareto_no_reasoning_merged"
TRAIN_DATA="data/pareto_true/pareto_and_near_pareto.jsonl"
EVAL_DATA="data/sft/designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"
OUTPUT_DIR="./grpo_output/no_reasoning_grpo_kl0.1"

# Reward 权重: compile 0.3 + sim 0.3 + rerank 0.3 = 0.9 (no format reward)
REWARD_COMPILE_WEIGHT=0.3
REWARD_SIM_WEIGHT=0.3
REWARD_GNN_WEIGHT=0.3
REWARD_FORMAT_WEIGHT=0.0

# 训练参数
BATCH_SIZE=2
NUM_GENERATIONS=4
EPOCHS=2
LR=5e-7
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
TEMPERATURE=0.9
KL_COEF=0.1
EVAL_STEPS=500
SAVE_STEPS=500
LOG_STEPS=10

# Uncertainty-aware proxy reward switching
MC_DROPOUT_PASSES=10          # M: number of stochastic forward passes
UNCERTAINTY_THRESHOLD=0.1     # tau_u: variance threshold to trigger real synthesis
REWARD_MODEL_UPDATE_STEPS=100 # K: fine-tune reward model every K training steps
REWARD_MODEL_FT_LR=2e-6      # learning rate for online reward model fine-tuning
REWARD_MODEL_FT_STEPS=50      # max gradient steps per reward model update

echo "=========================================="
echo "GRPO Training - Reasoning + Uncertainty-Aware Reward"
echo "=========================================="
echo "Base model: $BASE_MODEL"
echo "Train data: $TRAIN_DATA"
echo "Eval data:  $EVAL_DATA"
echo "Output:     $OUTPUT_DIR"
echo "Reward weights: compile=$REWARD_COMPILE_WEIGHT sim=$REWARD_SIM_WEIGHT gnn=$REWARD_GNN_WEIGHT fmt=$REWARD_FORMAT_WEIGHT"
echo "MC Dropout: M=$MC_DROPOUT_PASSES tau_u=$UNCERTAINTY_THRESHOLD"
echo "Reward model update: every $REWARD_MODEL_UPDATE_STEPS steps"
echo "=========================================="

export CUDA_VISIBLE_DEVICES=$GPU_IDS

python -m src.train_grpo \
    --train_data $TRAIN_DATA \
    --base_model $BASE_MODEL \
    --output_dir $OUTPUT_DIR \
    --reward_compile_weight $REWARD_COMPILE_WEIGHT \
    --reward_sim_weight $REWARD_SIM_WEIGHT \
    --reward_gnn_weight $REWARD_GNN_WEIGHT \
    --reward_format_weight $REWARD_FORMAT_WEIGHT \
    --batch_size $BATCH_SIZE \
    --num_generations $NUM_GENERATIONS \
    --epochs $EPOCHS \
    --lr $LR \
    --max_prompt_length $MAX_PROMPT_LENGTH \
    --max_new_tokens $MAX_NEW_TOKENS \
    --temperature $TEMPERATURE \
    --kl_coef $KL_COEF \
    --eval_data $EVAL_DATA \
    --eval_steps $EVAL_STEPS \
    --save_steps $SAVE_STEPS \
    --log_steps $LOG_STEPS \
    --mc_dropout_passes $MC_DROPOUT_PASSES \
    --uncertainty_threshold $UNCERTAINTY_THRESHOLD \
    --reward_model_update_steps $REWARD_MODEL_UPDATE_STEPS \
    --reward_model_finetune_lr $REWARD_MODEL_FT_LR \
    --reward_model_finetune_steps $REWARD_MODEL_FT_STEPS \
    --use_peft \
    --gpu $GPU_IDS

echo ""
echo "=========================================="
echo "GRPO Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="
