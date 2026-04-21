# HLS_code_generations

Code, training recipes, and evaluation benchmarks for generating HLS (High-Level Synthesis) C/C++ code with LLMs.

## Training Pipeline

![Training flow](fig/training_flow.pdf)

See [`fig/training_flow.pdf`](fig/training_flow.pdf) for the full training pipeline diagram (SFT → DPO → GRPO with reasoning + uncertainty-aware proxy reward switching).

## Directory Layout

```
HLS_code_generations/
├── training/                   # Training entry points (SFT / DPO / GRPO)
├── lib_function/               # Core library: dataset, scoring, inference, API client
├── proxy_comparative_model/    # Pairwise text ranker (proxy reward model) code
├── test/                       # Model evaluation scripts
├── forge_hls_testcase/         # 108-app held-out evaluation benchmark
├── hls-eval/                   # Upstream hls-eval benchmark (sharc-lab/hls-eval)
└── fig/                        # Figures (training flow, etc.)
```

### `training/` — Training scripts

| File | Description |
|---|---|
| `train_sft.py`         | Supervised Fine-Tuning on HLS code dataset (with assistant-only loss, optional PEFT/LoRA). |
| `train_dpo.py`         | Direct Preference Optimization on chosen/rejected HLS design pairs. |
| `train_grpo_trl.py`    | GRPO training via TRL, with reasoning-mode outputs (`<think>` + `<final_code>`) and testbench-based reward. |
| `run_train_sft.sh`     | Launch SFT with Accelerate (multi-GPU). |
| `run_sft_and_dpo.sh`   | Run SFT followed by DPO in a single pipeline. |
| `run_grpo_reasoning.sh`| Launch GRPO in reasoning mode with uncertainty-aware proxy reward switching. |

### `lib_function/` — Shared library

| File | Description |
|---|---|
| `config.py`                    | Global configuration (paths, model names, hyper-parameters). |
| `dataset.py`                   | JSONL loading and HuggingFace-Dataset construction with assistant-only label masking. |
| `preference_data.py`           | Chosen/rejected pair construction for DPO. |
| `scoring.py`                   | Reward / score functions used in preference data and RL rewards. |
| `ppa_evaluator.py`             | PPA (Performance / Power / Area) evaluator for synthesized designs. |
| `acceleration_utils.py`        | vLLM generation helpers for fast batched inference. |
| `inference.py`                 | Inference entry point for running a trained model on prompts. |
| `synthesize_reasoning_traces.py` | Offline augmentation: generate compact reasoning traces for SFT data. |
| `api_function.py`              | OpenAI-compatible API wrapper. Reads `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`) from env vars. |

### `proxy_comparative_model/` — Proxy reward model

| File | Description |
|---|---|
| `train_pairwise_text_ranker_unixcoder_hybrid.py` | Train a UniXcoder-based pairwise text ranker over HLS design pairs, used as the proxy reward model in GRPO. |
| `dataset_utils.py`                               | Graph / text dataset utilities feeding the pairwise ranker. |

### `test/` — Model evaluation

| File | Description |
|---|---|
| `test_sft.py`             | Evaluate SFT model pass-rate with the same testbench pipeline used in GRPO. |
| `test_local_models.py`    | Evaluate a local HuggingFace model on HLS benchmarks using vLLM. |
| `test_hls_eval_enhanced.py` | HLS-EVAL evaluation with enhanced prompts (header types + signatures) and `-I _tb_compat` compilation. |
| `test_gpt5_api.py`        | Evaluate a remote API model (e.g. GPT-5.1) on the 108-app prompt set. |

### `forge_hls_testcase/` — Held-out benchmark

The 108-app app-disjoint test set used throughout the paper. Apps are organized by source:

| Source | # Apps |
|---|---:|
| MachSuite | 4 |
| PolyBench | 6 |
| Vitis-HLS-Introductory-Examples-flatten | 8 |
| hls_algorithms | 22 |
| leetcode_hls_algorithms | 5 |
| operators | 29 |
| rosetta | 2 |
| rtl_chip | 4 |
| rtl_ip | 3 |
| rtl_module | 25 |
| **Total** | **108** |

Each app directory contains:
- **Source** (`*.c` / `*.cpp`) and **header** (`*.h` / `*.hpp`)
- **Testbench** (`*_cpp_testbench.cpp` / `*_c_testbench.c`)
- `design_0/` — a reference synthesized design (single pragma variant kept)
- `run_hls.tcl`, `top_function_name.txt` — HLS build metadata

### `hls-eval/` — Upstream benchmark

A copy of the public [sharc-lab/hls-eval](https://github.com/sharc-lab/hls-eval) benchmark, used by `test/test_hls_eval_enhanced.py` for cross-benchmark evaluation.

### `fig/` — Figures

| File | Description |
|---|---|
| `training_flow.pdf` | End-to-end training pipeline: SFT → DPO → GRPO (reasoning + uncertainty-gated reward). |

## Setup

### Environment variables

For scripts that call remote LLM APIs (`lib_function/api_function.py`, `test/test_gpt5_api.py`):

```bash
export OPENAI_API_KEY="<your key>"
export OPENAI_BASE_URL="<optional OpenAI-compatible endpoint>"
```

### Paths

All scripts use paths relative to their own location (`Path(__file__).resolve().parents[...]` in Python, `$(dirname "$0")` in shell). Input datasets are resolved relative to the project root; if running from a new checkout, place your jsonl datasets under a sibling `data/` folder or override with the `--input` / `--sft_data` / `--eval_data` arguments.

## Typical Workflows

**SFT**
```bash
bash training/run_train_sft.sh
```

**SFT → DPO**
```bash
bash training/run_sft_and_dpo.sh
```

**GRPO with reasoning + proxy reward**
```bash
bash training/run_grpo_reasoning.sh
```

**Evaluate a local / SFT model on the 108-app set**
```bash
python test/test_sft.py --model_path <path-to-checkpoint>
python test/test_local_models.py --model <hf-model-id-or-path>
```
