"""
Evaluate local HuggingFace models on HLS benchmarks using vLLM.
Reports Synthesizability and Functional Correctness at pass@1.

Supports both the 108-app test set and hls-eval benchmarks.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train_grpo import _parse_compile_and_sim_scores, load_prompt_jsonl, test_case

# ── hls-eval benchmark helpers (same as test_hls_eval_benchmarks.py) ──

HLS_EVAL_DATA_DIR = Path(__file__).resolve().parents[1] / "hls-eval/hls_eval_data"
COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 120

SYSTEM_HLS_GEN = (
    "You are an expert HLS engineer. Generate valid C/C++ HLS code that satisfies the "
    "functional description. Output only code. Do not include explanations."
)

MODELS = {
    "mashnoor/hls-vitis-ast-qwen-coder-7B": "mashnoor/hls-vitis-ast-qwen-coder-7B",
    "Qwen/Qwen2.5-Coder-7B-Instruct": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "Qwen/Qwen2.5-Coder-14B-Instruct": "Qwen/Qwen2.5-Coder-14B-Instruct",
    "bigcode/starcoder2-15b": "bigcode/starcoder2-15b",
    "deepseek-ai/deepseek-coder-6.7b-instruct": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "Qwen/Qwen2.5-Coder-32B-Instruct": "Qwen/Qwen2.5-Coder-32B-Instruct",
}


def extract_code(text: str) -> str:
    """Extract code from LLM response, handling markdown fences and tags."""
    if not text:
        return ""
    # Try markdown fences
    m = re.search(r"```(?:c|cpp|c\+\+|h)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Try <code> or <final_code> tags
    m = re.search(r"<(?:final_)?code>(.*?)</(?:final_)?code>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Return as-is (might be raw code)
    return text.strip()


def build_chat_messages(prompt: str, is_chat_model: bool):
    """Build messages for chat vs completion models."""
    if is_chat_model:
        return [
            {"role": "system", "content": SYSTEM_HLS_GEN},
            {"role": "user", "content": prompt},
        ]
    return None


def build_generation_prompt_108(prompt: str, metadata: Dict[str, Any]) -> str:
    top_function = str((metadata or {}).get("top_function", "")).strip()
    suffix = ""
    if top_function:
        suffix = f"\n\nThe top-level function name must be `{top_function}`."
    return f"{prompt.strip()}{suffix}\n\nReturn only the complete C/C++ source code."


# ── hls-eval benchmark functions ──

def load_training_apps(paths: List[str]) -> set:
    apps = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                m = obj.get("metadata", {})
                src = m.get("source_name", "").strip().lower()
                algo = m.get("algo_name", "").strip().lower()
                if src and algo:
                    apps.add(f"{src}/{algo}")
    return apps


def find_all_benchmarks(data_dir: Path) -> List[Dict[str, Any]]:
    benchmarks = []
    for suite_dir in sorted(data_dir.iterdir()):
        if not suite_dir.is_dir():
            continue
        suite = suite_dir.name
        for bench_dir in sorted(suite_dir.iterdir()):
            if not bench_dir.is_dir():
                continue
            config = bench_dir / "hls_eval_config.toml"
            if not config.exists():
                continue
            desc_file = bench_dir / "kernel_description.md"
            top_file = bench_dir / "top.txt"
            if not desc_file.exists() or not top_file.exists():
                continue
            tb_files = list(bench_dir.glob("*_tb.cpp")) + list(bench_dir.glob("*_tb.c"))
            if len(tb_files) != 1:
                continue
            kernel_files = [
                f for f in bench_dir.iterdir()
                if f.suffix in (".c", ".cpp", ".cc") and not f.stem.endswith("_tb")
            ]
            if len(kernel_files) != 1:
                continue
            h_files = [f for f in bench_dir.iterdir() if f.suffix in (".h", ".hpp", ".hh")]
            benchmarks.append({
                "suite": suite, "name": bench_dir.name, "key": f"{suite}/{bench_dir.name}",
                "dir": bench_dir,
                "description": desc_file.read_text(encoding="utf-8", errors="replace"),
                "top_function": top_file.read_text().strip(),
                "tb_file": tb_files[0], "kernel_file": kernel_files[0], "h_files": h_files,
            })
    return benchmarks


def normalize_key(key: str) -> str:
    return key.lower().replace("-", "_").replace(" ", "_")


def filter_benchmarks(benchmarks, train_apps):
    train_normalized = {normalize_key(a) for a in train_apps}
    filtered = []
    for b in benchmarks:
        key_norm = normalize_key(b["key"])
        suite_norm = b["suite"].lower()
        name_norm = b["name"].lower().replace("-", "_")
        matched = key_norm in train_normalized
        if not matched:
            for ta in train_normalized:
                parts = ta.split("/")
                if len(parts) == 2:
                    t_suite, t_algo = parts
                    if suite_norm.replace("_", "") == t_suite.replace("_", ""):
                        if name_norm == t_algo or name_norm.startswith(t_algo + "_") or t_algo.startswith(name_norm):
                            matched = True; break
                    if t_algo.replace("-", "_") == f"{suite_norm}_{name_norm}" or t_algo.replace("-", "_") == name_norm:
                        matched = True; break
        if not matched:
            filtered.append(b)
    return filtered


def compile_and_run_hls_eval(bench, candidate_code):
    tmp_dir = Path(tempfile.mkdtemp(prefix="hls_eval_tc_"))
    try:
        for f in bench["dir"].iterdir():
            if f.is_file():
                shutil.copy2(f, tmp_dir / f.name)
        kernel_name = bench["kernel_file"].name
        (tmp_dir / kernel_name).write_text(candidate_code, encoding="utf-8")
        tb_name = bench["tb_file"].name
        tb_path = tmp_dir / tb_name
        exe_path = tmp_dir / "candidate.out"
        if tb_path.suffix == ".cpp" or bench["kernel_file"].suffix == ".cpp":
            cmd = ["g++", "-std=gnu++17", "-O2", "-w"]
        else:
            cmd = ["gcc", "-std=gnu11", "-O2", "-w"]
        cmd.extend(["-I", str(tmp_dir), str(tb_path), str(tmp_dir / kernel_name)])
        cmd.extend(["-lm", "-lstdc++", "-o", str(exe_path)])
        try:
            cp = subprocess.run(cmd, capture_output=True, text=True, timeout=COMPILE_TIMEOUT)
        except subprocess.TimeoutExpired:
            return {"pass": False, "compiled": False, "detail": "compile_timeout"}
        if cp.returncode != 0:
            return {"pass": False, "compiled": False, "detail": "compile_fail", "compile_stderr": cp.stderr[-500:]}
        try:
            rp = subprocess.run(
                ["bash", "-lc", f"ulimit -s unlimited; cd '{tmp_dir}' && '{exe_path}'"],
                capture_output=True, text=True, timeout=RUN_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return {"pass": False, "compiled": True, "detail": "run_timeout"}
        stdout = rp.stdout.strip()
        last_line = stdout.splitlines()[-1].strip().lower() if stdout.splitlines() else ""
        passed = rp.returncode == 0 and any(kw in last_line for kw in ["pass", "success", "correct"])
        if not passed and rp.returncode == 0 and "fail" not in last_line and "error" not in last_line:
            passed = True
        return {"pass": passed, "compiled": True, "detail": "pass" if passed else "functional_fail"}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate local models on HLS benchmarks")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--benchmark", type=str, choices=["108app", "hls_eval", "both"], default="both")
    parser.add_argument(
        "--test_data", type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/sft/designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"),
    )
    parser.add_argument(
        "--train_files", type=str, nargs="+",
        default=[str(Path(__file__).resolve().parents[2] / "data/pareto_true/pareto_and_near_pareto.jsonl")],
    )
    parser.add_argument("--num_generations", type=int, default=1, help="Generations per sample")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "GPT5_eval/local_models"))
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    model_short = args.model.split("/")[-1]
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{model_short}_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model} (tp={args.tensor_parallel_size}) ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        dtype="auto",
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    tokenizer = llm.get_tokenizer()
    is_chat = hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    print(f"Model loaded. Chat model: {is_chat}")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_generations,
    )

    def generate_batch(prompts_or_messages: list, is_chat: bool) -> List[List[str]]:
        """Generate responses for a batch of prompts. Returns list of list of responses."""
        if is_chat:
            outputs = llm.chat(prompts_or_messages, sampling_params=sampling_params)
        else:
            # For completion models, format prompt with system instruction
            formatted = []
            for p in prompts_or_messages:
                formatted.append(f"{SYSTEM_HLS_GEN}\n\n{p}")
            outputs = llm.generate(formatted, sampling_params=sampling_params)
        results = []
        for output in outputs:
            gen_texts = [o.text for o in output.outputs]
            results.append(gen_texts)
        return results

    all_results = {}

    # ── 108-app benchmark ──
    if args.benchmark in ("108app", "both"):
        print("\n" + "=" * 60)
        print("108-APP BENCHMARK")
        print("=" * 60)
        data = load_prompt_jsonl(args.test_data)
        if args.max_samples > 0:
            data = data[: args.max_samples]
        print(f"Loaded {len(data)} samples")

        # Build all prompts
        prompts_108 = []
        for sample in data:
            p = build_generation_prompt_108(sample["prompt"], sample.get("metadata", {}))
            prompts_108.append(p)

        # Build chat messages or raw prompts
        if is_chat:
            chat_msgs = [
                [{"role": "system", "content": SYSTEM_HLS_GEN}, {"role": "user", "content": p}]
                for p in prompts_108
            ]
            batch_input = chat_msgs
        else:
            batch_input = prompts_108

        print(f"Generating {len(batch_input)} x {args.num_generations} responses...")
        all_responses = generate_batch(batch_input, is_chat)

        # Evaluate
        total = 0
        compile_pass = 0
        func_pass = 0
        fail_counter = Counter()
        per_sample_compile = []
        per_sample_func = []

        for idx, (sample, responses) in enumerate(zip(data, all_responses)):
            metadata = sample.get("metadata", {})
            sample_compile_flags = []
            sample_func_flags = []

            for gen_idx, resp in enumerate(responses):
                code = extract_code(resp)
                tc_out = test_case(sample["prompt"], code, metadata)
                passed, comp_score, sim_score = _parse_compile_and_sim_scores(tc_out)
                compiled = comp_score > 0.0

                sample_compile_flags.append(compiled)
                sample_func_flags.append(bool(passed))
                total += 1
                compile_pass += int(compiled)
                func_pass += int(passed)
                if not passed:
                    detail = tc_out.get("detail", "unknown") if isinstance(tc_out, dict) else "unknown"
                    fail_counter[detail] += 1

                # Save debug
                dbg = output_dir / "108app" / f"{idx:04d}" / f"gen_{gen_idx}"
                dbg.mkdir(parents=True, exist_ok=True)
                (dbg / "response.txt").write_text(resp or "", encoding="utf-8")
                (dbg / "code.txt").write_text(code or "", encoding="utf-8")
                with open(dbg / "result.json", "w") as f:
                    json.dump(tc_out if isinstance(tc_out, dict) else {"raw": str(tc_out)}, f, indent=2)

            per_sample_compile.append(sample_compile_flags)
            per_sample_func.append(sample_func_flags)

            if (idx + 1) % 20 == 0:
                print(f"  [{idx+1}/{len(data)}] synth={compile_pass}/{total} func={func_pass}/{total}")

        # Compute pass@k
        n_prompts = len(data)
        pass_k_values = [1, 5, 10] if args.num_generations >= 10 else [1]
        summary_108 = {
            "model": args.model, "benchmark": "108app",
            "num_prompts": n_prompts, "num_generations": args.num_generations,
            "candidate_compile_rate": compile_pass / max(total, 1),
            "candidate_func_rate": func_pass / max(total, 1),
        }
        for k in pass_k_values:
            if k > args.num_generations:
                continue
            synth_k = sum(1 for flags in per_sample_compile if any(flags[:k]))
            func_k = sum(1 for flags in per_sample_func if any(flags[:k]))
            summary_108[f"synthesizability_pass@{k}"] = synth_k / max(n_prompts, 1)
            summary_108[f"functional_correctness_pass@{k}"] = func_k / max(n_prompts, 1)

        summary_108["fail_details"] = dict(fail_counter.most_common(20))

        print(f"\n108-APP Results ({args.model}):")
        for k in pass_k_values:
            if f"synthesizability_pass@{k}" in summary_108:
                print(f"  Synth pass@{k} = {summary_108[f'synthesizability_pass@{k}']:.4f}")
                print(f"  Func  pass@{k} = {summary_108[f'functional_correctness_pass@{k}']:.4f}")

        with open(output_dir / "108app_results.json", "w") as f:
            json.dump(summary_108, f, indent=2)
        all_results["108app"] = summary_108

    # ── hls-eval benchmark ──
    if args.benchmark in ("hls_eval", "both"):
        print("\n" + "=" * 60)
        print("HLS-EVAL BENCHMARK")
        print("=" * 60)
        train_apps = load_training_apps(args.train_files)
        all_bench = find_all_benchmarks(HLS_EVAL_DATA_DIR)
        benchmarks = filter_benchmarks(all_bench, train_apps)
        if args.max_samples > 0:
            benchmarks = benchmarks[: args.max_samples]
        print(f"Benchmarks: {len(benchmarks)} (filtered from {len(all_bench)})")

        # Build prompts
        bench_prompts = []
        for b in benchmarks:
            p = f"{b['description'].strip()}\n\nThe top-level function name must be `{b['top_function']}`.\n\nReturn only the complete C/C++ source code."
            bench_prompts.append(p)

        if is_chat:
            chat_msgs = [
                [{"role": "system", "content": SYSTEM_HLS_GEN}, {"role": "user", "content": p}]
                for p in bench_prompts
            ]
            batch_input = chat_msgs
        else:
            batch_input = bench_prompts

        print(f"Generating {len(batch_input)} responses...")
        all_responses = generate_batch(batch_input, is_chat)

        total = 0
        compile_pass = 0
        func_pass = 0
        fail_counter = Counter()

        for idx, (bench, responses) in enumerate(zip(benchmarks, all_responses)):
            for gen_idx, resp in enumerate(responses):
                code = extract_code(resp)
                result = compile_and_run_hls_eval(bench, code)

                total += 1
                compiled = result.get("compiled", False)
                passed = result.get("pass", False)
                compile_pass += int(compiled)
                func_pass += int(passed)
                if not passed:
                    fail_counter[result.get("detail", "unknown")] += 1

                dbg = output_dir / "hls_eval" / f"{idx:04d}_{bench['key'].replace('/', '_')}" / f"gen_{gen_idx}"
                dbg.mkdir(parents=True, exist_ok=True)
                (dbg / "response.txt").write_text(resp or "", encoding="utf-8")
                (dbg / "code.txt").write_text(code or "", encoding="utf-8")
                with open(dbg / "result.json", "w") as f:
                    json.dump(result, f, indent=2)

            if (idx + 1) % 10 == 0:
                print(f"  [{idx+1}/{len(benchmarks)}] synth={compile_pass}/{total} func={func_pass}/{total}")

        summary_hls = {
            "model": args.model, "benchmark": "hls_eval",
            "num_benchmarks": len(benchmarks), "num_generations": args.num_generations,
            "synthesizability_pass@1": compile_pass / max(total, 1),
            "functional_correctness_pass@1": func_pass / max(total, 1),
            "compile_pass": compile_pass, "func_pass": func_pass,
            "fail_details": dict(fail_counter.most_common(20)),
        }

        print(f"\nHLS-EVAL Results ({args.model}):")
        print(f"  Synth pass@1 = {summary_hls['synthesizability_pass@1']:.4f} ({compile_pass}/{total})")
        print(f"  Func  pass@1 = {summary_hls['functional_correctness_pass@1']:.4f} ({func_pass}/{total})")

        with open(output_dir / "hls_eval_results.json", "w") as f:
            json.dump(summary_hls, f, indent=2)
        all_results["hls_eval"] = summary_hls

    # Final combined summary
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {output_dir}")


if __name__ == "__main__":
    main()
