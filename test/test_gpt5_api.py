"""
Evaluate GPT-5.1 API generations on HLS prompt JSONL.
Reports Synthesizability (compilation) and Functional Correctness
at pass@1, pass@5, pass@10.
"""

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api_function import llm_call
from src.train_grpo import _parse_compile_and_sim_scores, load_prompt_jsonl, test_case


SYSTEM_HLS_GEN = (
    "You are an expert HLS engineer. Generate valid C/C++ HLS code that satisfies the "
    "functional description. Output only code. Do not include explanations."
)


def batched(seq: List[Dict[str, Any]], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def build_generation_prompt(prompt: str, metadata: Dict[str, Any]) -> str:
    top_function = str((metadata or {}).get("top_function", "")).strip()
    suffix = ""
    if top_function:
        suffix = f"\n\nThe top-level function name must be `{top_function}`."
    return f"{prompt.strip()}{suffix}\n\nReturn only the complete C/C++ source code."


def dump_candidate(
    debug_root: Path,
    batch_idx: int,
    sample_idx: int,
    candidate_idx: int,
    prompt: str,
    generated_text: str,
    tc_out: Dict[str, Any],
) -> None:
    candidate_dir = (
        debug_root
        / f"batch_{batch_idx:04d}"
        / f"sample_{sample_idx:04d}"
        / f"candidate_{candidate_idx:02d}"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)
    (candidate_dir / "prompt.txt").write_text(prompt or "", encoding="utf-8")
    (candidate_dir / "response.txt").write_text(generated_text or "", encoding="utf-8")
    with open(candidate_dir / "test_case_result.json", "w", encoding="utf-8") as f:
        json.dump(tc_out, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate GPT-5.1 API on HLS testcase JSONL")
    parser.add_argument(
        "--test_data",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/sft/designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"),
        help="Prompt JSONL with metadata.file_path for testcase evaluation",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all")
    parser.add_argument("--batch_size", type=int, default=1, help="Logical prompt batch size for progress display")
    parser.add_argument("--num_generations", type=int, default=10, help="k for pass@k")
    parser.add_argument("--pass_k_values", type=str, default="1,5,10", help="Comma-separated pass@k values to report")
    parser.add_argument("--sleep_sec", type=float, default=0.0, help="Sleep between API calls")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--output_json", type=str, default=None, help="Save summary JSON")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Save per-candidate results JSONL")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="Model name for the API")
    parser.add_argument(
        "--debug_output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "GPT5_eval"),
        help="Directory to save prompts/responses/testcase results",
    )
    args = parser.parse_args()
    pass_k_values = sorted({int(x.strip()) for x in args.pass_k_values.split(",") if x.strip()})
    if not pass_k_values or min(pass_k_values) <= 0:
        raise ValueError(f"Invalid --pass_k_values: {args.pass_k_values}")
    if args.num_generations < max(pass_k_values):
        args.num_generations = max(pass_k_values)

    data = load_prompt_jsonl(args.test_data)
    if not data:
        raise ValueError(f"No valid samples found in {args.test_data}")
    if args.max_samples > 0:
        data = data[: args.max_samples]

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_run_dir = Path(args.debug_output_dir) / f"run_{run_tag}"
    debug_run_dir.mkdir(parents=True, exist_ok=True)

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_jsonl).write_text("", encoding="utf-8")

    candidate_total = 0
    candidate_compile_pass = 0
    candidate_func_pass = 0
    prompt_total = 0
    synth_pass_at_k = {k: 0 for k in pass_k_values}
    func_pass_at_k = {k: 0 for k in pass_k_values}
    fail_detail_counter: Counter = Counter()

    for bidx, batch in enumerate(batched(data, args.batch_size), start=1):
        for sample_idx, sample in enumerate(batch):
            prompt = sample["prompt"]
            metadata = sample.get("metadata", {})
            api_prompt = build_generation_prompt(prompt, metadata)
            sample_compile_flags: List[bool] = []
            sample_func_flags: List[bool] = []

            for candidate_idx in range(args.num_generations):
                generated_text = llm_call(api_prompt, system=SYSTEM_HLS_GEN, model=args.model)
                tc_out = test_case(prompt, generated_text, metadata)
                passed, compile_score, sim_score = _parse_compile_and_sim_scores(tc_out)
                compiled = compile_score > 0.0
                sample_compile_flags.append(compiled)
                sample_func_flags.append(bool(passed))

                dump_candidate(
                    debug_root=debug_run_dir,
                    batch_idx=bidx,
                    sample_idx=sample_idx,
                    candidate_idx=candidate_idx,
                    prompt=api_prompt,
                    generated_text=generated_text,
                    tc_out=tc_out if isinstance(tc_out, dict) else {"raw": str(tc_out)},
                )

                candidate_total += 1
                candidate_compile_pass += int(compiled)
                candidate_func_pass += int(passed)
                if not passed:
                    detail = str(tc_out.get("detail", "unknown")) if isinstance(tc_out, dict) else "unknown"
                    fail_detail_counter[detail] += 1

                if args.output_jsonl:
                    rec = {
                        "batch": bidx,
                        "sample_index": sample_idx,
                        "candidate_index": candidate_idx,
                        "compiled": compiled,
                        "passed": bool(passed),
                        "compile_score": float(compile_score),
                        "sim_score": float(sim_score),
                        "detail": tc_out.get("detail", "unknown") if isinstance(tc_out, dict) else "unknown",
                        "file_path": metadata.get("file_path"),
                        "top_function": metadata.get("top_function"),
                    }
                    with open(args.output_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)

            for k in pass_k_values:
                if any(sample_compile_flags[:k]):
                    synth_pass_at_k[k] += 1
                if any(sample_func_flags[:k]):
                    func_pass_at_k[k] += 1

        prompt_total += len(batch)

        if bidx % args.log_every == 0:
            synth_k_text = " ".join(
                f"synth_pass@{k}={synth_pass_at_k[k] / max(prompt_total, 1):.4f}" for k in pass_k_values
            )
            func_k_text = " ".join(
                f"func_pass@{k}={func_pass_at_k[k] / max(prompt_total, 1):.4f}" for k in pass_k_values
            )
            print(
                f"[batch {bidx}] prompts={prompt_total} "
                f"compile_rate={candidate_compile_pass / max(candidate_total, 1):.4f} "
                f"func_rate={candidate_func_pass / max(candidate_total, 1):.4f}\n"
                f"  {synth_k_text}\n"
                f"  {func_k_text}"
            )

    # Final summary
    summary = {
        "model": args.model,
        "test_data": args.test_data,
        "debug_output_dir": str(debug_run_dir),
        "num_prompts": prompt_total,
        "num_generations": args.num_generations,
        "reported_pass_k": pass_k_values,
        "num_candidates": candidate_total,
        "candidate_compile_rate": candidate_compile_pass / max(candidate_total, 1),
        "candidate_func_pass_rate": candidate_func_pass / max(candidate_total, 1),
        "fail_details_top20": dict(fail_detail_counter.most_common(20)),
    }
    for k in pass_k_values:
        summary[f"synthesizability_pass@{k}"] = synth_pass_at_k[k] / max(prompt_total, 1)
        summary[f"functional_correctness_pass@{k}"] = func_pass_at_k[k] / max(prompt_total, 1)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Prompts: {prompt_total}, Generations per prompt: {args.num_generations}")
    print(f"\nSynthesizability (Compilation Success):")
    for k in pass_k_values:
        print(f"  pass@{k} = {synth_pass_at_k[k] / max(prompt_total, 1):.4f} ({synth_pass_at_k[k]}/{prompt_total})")
    print(f"\nFunctional Correctness:")
    for k in pass_k_values:
        print(f"  pass@{k} = {func_pass_at_k[k] / max(prompt_total, 1):.4f} ({func_pass_at_k[k]}/{prompt_total})")
    print("=" * 60)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
