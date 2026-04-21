"""
Evaluate SFT model pass rate with the same testbench pipeline used in train_grpo.py.
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except Exception:
    PeftModel = None

try:
    from .train_grpo import (
        TB_COMPAT_DIR,
        _extract_code_from_generation,
        _find_testbench_path,
        _format_prompt,
        _format_compile_stderr,
        _parse_test_case_result,
        _resolve_repo_path,
        load_prompt_jsonl,
    )
except ImportError:
    from train_grpo import (
        TB_COMPAT_DIR,
        _extract_code_from_generation,
        _find_testbench_path,
        _format_prompt,
        _format_compile_stderr,
        _parse_test_case_result,
        _resolve_repo_path,
        load_prompt_jsonl,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TC_TIMEOUT_SEC = int(os.environ.get("TEST_SFT_TIMEOUT_SEC", os.environ.get("GRPO_TEST_TIMEOUT_SEC", "20")))
DEFAULT_TC_COMPILE_TIMEOUT_SEC = int(
    os.environ.get("TEST_SFT_COMPILE_TIMEOUT_SEC", os.environ.get("GRPO_COMPILE_TIMEOUT_SEC", "20"))
)
DEFAULT_TC_TMP_ROOT = Path(
    os.environ.get("TEST_SFT_TMP_ROOT", str(PROJECT_ROOT / "logs" / "sft_eval"))
).resolve()


def load_model_and_tokenizer(model_path: str):
    model_dir = Path(model_path)
    is_adapter = (model_dir / "adapter_config.json").exists()

    def resolve_local_hf_snapshot(model_ref: str) -> str:
        candidate = Path(model_ref)
        if candidate.exists():
            return str(candidate)

        hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub_dir = hf_home / "hub" / f"models--{model_ref.replace('/', '--')}"
        if not hub_dir.exists():
            return model_ref

        ref_file = hub_dir / "refs" / "main"
        if ref_file.exists():
            revision = ref_file.read_text(encoding="utf-8").strip()
            snapshot_dir = hub_dir / "snapshots" / revision
            if snapshot_dir.exists():
                return str(snapshot_dir)

        snapshots_dir = hub_dir / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
            if snapshots:
                return str(snapshots[-1])

        return model_ref

    # Evaluation should work against already-downloaded checkpoints without
    # issuing Hub requests, which can fail in restricted environments.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    model_source = str(model_dir)
    base_load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }

    tokenizer = AutoTokenizer.from_pretrained(model_source, **base_load_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    if is_adapter:
        if PeftModel is None:
            raise ImportError("peft is required to load adapter checkpoints")
        with open(model_dir / "adapter_config.json", "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model = adapter_cfg.get("base_model_name_or_path")
        if not base_model:
            raise ValueError(f"adapter_config.json missing base_model_name_or_path: {model_dir / 'adapter_config.json'}")
        resolved_base_model = resolve_local_hf_snapshot(base_model)
        base = AutoModelForCausalLM.from_pretrained(
            resolved_base_model,
            dtype=model_dtype,
            device_map="auto",
            **base_load_kwargs,
        )
        model = PeftModel.from_pretrained(base, model_source, local_files_only=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_source,
            dtype=model_dtype,
            device_map="auto",
            **base_load_kwargs,
        )

    model.eval()
    return model, tokenizer


def batched(seq: List[Dict], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def dump_candidate_debug_artifacts(
    debug_root: Path,
    batch_idx: int,
    sample_idx: int,
    candidate_idx: int,
    prompt: str,
    generated_text: str,
    metadata: Dict[str, Any],
    tc_out: Dict[str, Any],
    passed: bool,
    tc_score: float,
) -> Path:
    candidate_dir = (
        debug_root
        / f"batch_{batch_idx:04d}"
        / f"sample_{sample_idx:04d}"
        / f"candidate_{candidate_idx:02d}"
    )
    candidate_dir.mkdir(parents=True, exist_ok=True)

    (candidate_dir / "prompt.txt").write_text(prompt or "", encoding="utf-8")
    (candidate_dir / "model_output.log").write_text(generated_text or "", encoding="utf-8")

    source_path = _resolve_repo_path(str(metadata.get("file_path", "")))
    candidate_ext = source_path.suffix if source_path and source_path.suffix in {".c", ".cpp", ".cc", ".cxx"} else ".c"
    candidate_code = _extract_code_from_generation(generated_text)
    candidate_filename = f"candidate{candidate_ext}"
    (candidate_dir / candidate_filename).write_text(candidate_code or "", encoding="utf-8")

    compile_cmd = ""
    compile_stderr = ""
    if isinstance(tc_out, dict):
        compile_cmd = str(tc_out.get("compile_cmd", "") or "")
        compile_stderr = str(tc_out.get("compile_stderr", "") or "")

    # Build local repro artifacts in the candidate directory:
    # g++ candidate.cpp candidate_tb.cpp -I ...
    local_compile_cmd = ""
    if source_path is not None:
        tb_path = _find_testbench_path(source_path, metadata or {})
        if tb_path is not None and tb_path.exists():
            tb_suffix = tb_path.suffix if tb_path.suffix in {".c", ".cpp", ".cc", ".cxx"} else ".cpp"
            tb_name = f"candidate_tb{tb_suffix}"
            shutil.copyfile(tb_path, candidate_dir / tb_name)

            use_cpp = candidate_ext in {".cpp", ".cc", ".cxx"} or tb_suffix in {".cpp", ".cc", ".cxx"}
            if use_cpp:
                cmd = ["g++", "-std=gnu++17", "-O2", "-w"]
            else:
                cmd = ["gcc", "-std=gnu11", "-O2", "-w"]
            if TB_COMPAT_DIR.exists():
                cmd.extend(["-I", str(TB_COMPAT_DIR)])
            cmd.extend(["-I", str(tb_path.parent), candidate_filename, tb_name, "-lm", "-o", "candidate.out"])
            local_compile_cmd = shlex.join(cmd)

    # Persist compile command as standalone artifacts for easy repro.
    (candidate_dir / "compile_command_original.txt").write_text(compile_cmd, encoding="utf-8")
    (candidate_dir / "compile_command.txt").write_text(local_compile_cmd or compile_cmd, encoding="utf-8")
    active_cmd = local_compile_cmd or compile_cmd
    if active_cmd:
        compile_sh = (
            "#!/usr/bin/env bash\nset -euo pipefail\n\n"
            "cd \"$(dirname \"$0\")\"\n\n"
            + active_cmd.strip()
            + "\n"
        )
        compile_sh_path = candidate_dir / "compile_command.sh"
        compile_sh_path.write_text(compile_sh, encoding="utf-8")
        os.chmod(compile_sh_path, 0o755)

    if compile_stderr:
        (candidate_dir / "compile_stderr.log").write_text(compile_stderr, encoding="utf-8")

    with open(candidate_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata or {}, f, ensure_ascii=False, indent=2)
    with open(candidate_dir / "test_case_result.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "passed": bool(passed),
                "testcase_score": float(tc_score),
                "detail": tc_out.get("detail", "unknown"),
                "test_case_result": tc_out,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return candidate_dir


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _testbench_uses_candidate_arg_mode(tb_text: str) -> bool:
    text = tb_text or ""
    if not (re.search(r"\bargc\b", text) and re.search(r"\bargv\b", text)):
        return False
    if re.search(r"candidate\s*=\s*\(argc\s*>\s*1\)", text):
        return True
    return any(token in text for token in ("dlopen(", "build_candidate_shared_lib", "system(cmd)"))


def _compile_and_run_testbench(
    source_path: Path,
    testbench_path: Path,
    candidate_code: str,
    timeout_sec: int,
    compile_timeout_sec: int,
) -> Dict[str, Any]:
    tb_text = testbench_path.read_text(encoding="utf-8", errors="ignore")
    candidate_arg_mode = _testbench_uses_candidate_arg_mode(tb_text)

    def _compile_tb(
        tb_file: Path,
        include_dir: Path,
        exe_file: Path,
        extra_sources: List[Path] | None = None,
    ) -> Tuple[subprocess.CompletedProcess[str], str]:
        use_cpp = tb_file.suffix in {".cpp", ".cc", ".cxx"}
        if use_cpp:
            cmd = ["g++", "-std=gnu++17", "-O2", "-fpermissive", "-w"]
        else:
            cmd = ["gcc", "-std=gnu11", "-O2", "-w"]
        if TB_COMPAT_DIR.exists():
            cmd.extend(["-I", str(TB_COMPAT_DIR)])
        cmd.extend(["-I", str(include_dir), str(tb_file)])
        if extra_sources:
            cmd.extend([str(x) for x in extra_sources])
        if "dlopen(" in tb_text or "dlsym(" in tb_text:
            cmd.append("-ldl")
        cmd.extend(["-lm", "-o", str(exe_file)])
        cmd_str = shlex.join(cmd)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=compile_timeout_sec,
        )
        return proc, cmd_str

    fixed_tmp_dir_env = os.environ.get("GRPO_TC_FIXED_DIR", "").strip()
    if fixed_tmp_dir_env:
        tmp_dir = Path(fixed_tmp_dir_env).resolve()
        tmp_dir.mkdir(parents=True, exist_ok=True)
        exe_path = tmp_dir / "candidate.out"
        cleanup_tmp = False
    else:
        DEFAULT_TC_TMP_ROOT.mkdir(parents=True, exist_ok=True)
        tmp_dir = Path(tempfile.mkdtemp(prefix="sft_tc_", dir=str(DEFAULT_TC_TMP_ROOT)))
        exe_path = tmp_dir / "candidate.out"
        cleanup_tmp = True

    try:
        compile_cmd_str = ""
        try:
            if candidate_arg_mode:
                candidate_ext = source_path.suffix if source_path.suffix in {".c", ".cpp", ".cc", ".cxx"} else ".c"
                candidate_path = tmp_dir / f"candidate{candidate_ext}"
                _write_text(candidate_path, candidate_code)
                compile_proc, compile_cmd_str = _compile_tb(testbench_path, testbench_path.parent, exe_path)
                run_cmd = [str(exe_path), str(candidate_path)]
            else:
                staged_kernel_dir = tmp_dir / "kernel"
                shutil.copytree(source_path.parent, staged_kernel_dir, dirs_exist_ok=True)
                staged_source = staged_kernel_dir / source_path.name
                staged_tb = staged_kernel_dir / testbench_path.name
                if not staged_tb.exists():
                    return {
                        "pass": False,
                        "score": 0.0,
                        "detail": "testbench_not_found_after_stage",
                        "tmp_dir": str(tmp_dir),
                    }
                _write_text(staged_source, candidate_code)
                compile_proc, compile_cmd_str = _compile_tb(staged_tb, staged_kernel_dir, exe_path)
                run_cmd = [str(exe_path)]
        except subprocess.TimeoutExpired:
            return {
                "pass": False,
                "score": 0.0,
                "detail": "compile_timeout",
                "tmp_dir": str(tmp_dir),
            }

        _write_text(tmp_dir / "compile_cmd.log", compile_cmd_str + "\n")
        _write_text(tmp_dir / "compile_stdout.log", compile_proc.stdout or "")
        _write_text(tmp_dir / "compile_stderr.log", compile_proc.stderr or "")

        if compile_proc.returncode != 0:
            return {
                "pass": False,
                "score": 0.0,
                "detail": "compile_fail",
                "compile_cmd": compile_cmd_str,
                "compile_stderr": _format_compile_stderr(compile_proc.stderr),
                "tmp_dir": str(tmp_dir),
            }

        try:
            _write_text(tmp_dir / "run_cmd.log", shlex.join(run_cmd) + "\n")
            run_proc = subprocess.run(
                run_cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                cwd=str(source_path.parent if candidate_arg_mode else staged_kernel_dir),
            )
        except subprocess.TimeoutExpired:
            return {
                "pass": False,
                "score": 0.0,
                "detail": "timeout",
                "compile_cmd": compile_cmd_str,
                "tmp_dir": str(tmp_dir),
            }

        _write_text(tmp_dir / "run_stdout.log", run_proc.stdout or "")
        _write_text(tmp_dir / "run_stderr.log", run_proc.stderr or "")

        out_lines = [x.strip().lower() for x in run_proc.stdout.splitlines() if x.strip()]
        passed = run_proc.returncode == 0 and any(line == "pass" for line in out_lines[-5:])
        if passed:
            return {
                "pass": True,
                "score": 1.0,
                "detail": "pass",
                "compile_cmd": compile_cmd_str,
                "tmp_dir": str(tmp_dir),
            }

        if any(line == "fail" for line in out_lines[-5:]):
            detail = "functional_fail"
        else:
            detail = f"runtime_nonpass_exit_{run_proc.returncode}"
        return {
            "pass": False,
            "score": 0.0,
            "detail": detail,
            "compile_cmd": compile_cmd_str,
            "run_exit_code": int(run_proc.returncode),
            "stdout_tail": run_proc.stdout[-1200:],
            "stderr_tail": run_proc.stderr[-1200:],
            "tmp_dir": str(tmp_dir),
        }
    finally:
        if cleanup_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def evaluate_test_case(prompt: str, generated_code: str, metadata: Dict[str, Any], reasoning_mode: bool = False) -> Dict[str, Any]:
    code = _extract_code_from_generation(generated_code, reasoning_mode=reasoning_mode)
    if not code:
        return {"pass": False, "score": 0.0, "detail": "empty_generation"}

    source_path = _resolve_repo_path(str(metadata.get("file_path", "")))
    if source_path is None:
        return {"pass": False, "score": 0.0, "detail": "missing_or_invalid_file_path"}

    testbench_path = _find_testbench_path(source_path, metadata)
    if testbench_path is None:
        return {"pass": False, "score": 0.0, "detail": "testbench_not_found"}

    timeout_sec = int(metadata.get("timeout_sec", DEFAULT_TC_TIMEOUT_SEC))
    compile_timeout_sec = int(metadata.get("compile_timeout_sec", DEFAULT_TC_COMPILE_TIMEOUT_SEC))
    return _compile_and_run_testbench(
        source_path=source_path,
        testbench_path=testbench_path,
        candidate_code=code,
        timeout_sec=timeout_sec,
        compile_timeout_sec=compile_timeout_sec,
    )


def generate_batch(
    model,
    tokenizer,
    prompts: List[str],
    num_generations: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[List[str], List[str]]:
    formatted = [_format_prompt(p) for p in prompts]
    encoded = tokenizer(
        formatted,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    model_device = next(model.parameters()).device
    encoded = {k: v.to(model_device) for k, v in encoded.items()}

    prompt_lens = encoded["attention_mask"].sum(dim=1)
    with torch.no_grad():
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_generations,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    repeated_prompt_lens = prompt_lens.repeat_interleave(num_generations)
    repeated_prompts: List[str] = []
    responses: List[str] = []
    for i in range(generated.size(0)):
        prompt_idx = i // num_generations
        start = int(repeated_prompt_lens[i].item())
        completion_ids = generated[i, start:]
        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip()
        repeated_prompts.append(prompts[prompt_idx])
        responses.append(text)
    return repeated_prompts, responses


def main():
    parser = argparse.ArgumentParser(description="Evaluate SFT model pass rate")
    parser.add_argument("--model_path", type=str, required=True, help="SFT model checkpoint path")
    parser.add_argument(
        "--test_data",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data/sft/designs_with_metrics_app_disjoint_test_108_apps_min_worst_latency_per_app.jsonl"),
        help="Prompt/SFT JSONL for testbench evaluation; SFT records can omit metadata.file_path",
    )
    parser.add_argument("--max_samples", type=int, default=100, help="0 means all")
    parser.add_argument("--batch_size", type=int, default=2, help="Prompt batch size")
    parser.add_argument("--num_generations", type=int, default=4, help="k for pass@k")
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--gpu", type=str, default=None, help="GPU devices, e.g. '0' or '0,1'")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--output_json", type=str, default=None, help="Save summary JSON")
    parser.add_argument("--output_jsonl", type=str, default=None, help="Save per-candidate results JSONL")
    parser.add_argument(
        "--debug_output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "sft_output/sft_test"),
        help="Directory to save model outputs and generated candidate files for debugging",
    )
    parser.add_argument("--timeout_sec", type=int, default=DEFAULT_TC_TIMEOUT_SEC, help="Per-testbench run timeout")
    parser.add_argument(
        "--compile_timeout_sec",
        type=int,
        default=DEFAULT_TC_COMPILE_TIMEOUT_SEC,
        help="Per-testbench compile timeout",
    )
    parser.add_argument(
        "--reasoning_mode",
        action="store_true",
        help="Enable reasoning mode: extract code from <final_code> tags instead of raw output",
    )
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"Using GPU(s): {args.gpu}")

    data = load_prompt_jsonl(args.test_data)
    if not data:
        raise ValueError(f"No valid samples found in {args.test_data}")
    if args.max_samples > 0:
        data = data[: args.max_samples]

    print(f"Loaded {len(data)} prompts from {args.test_data}")
    print(f"Evaluating with num_generations={args.num_generations} (pass@{args.num_generations})")

    model, tokenizer = load_model_and_tokenizer(args.model_path)
    print(f"Model loaded: {args.model_path}")
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_run_dir = Path(args.debug_output_dir) / f"run_{run_tag}"
    debug_run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Debug artifacts dir: {debug_run_dir}")

    with open(debug_run_dir / "run_args.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    candidate_total = 0
    candidate_pass = 0
    prompt_total = 0
    prompt_pass_at_k = 0
    testcase_scores: List[float] = []
    fail_detail_counter: Counter = Counter()

    if args.output_jsonl:
        Path(args.output_jsonl).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            f.write("")

    for bidx, batch in enumerate(batched(data, args.batch_size), start=1):
        prompts = [x["prompt"] for x in batch]
        metadata_list = [x.get("metadata", {}) for x in batch]

        repeated_prompts, responses = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            num_generations=args.num_generations,
            max_prompt_length=args.max_prompt_length,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        for i, gen_text in enumerate(responses):
            group_idx = i // args.num_generations
            cand_idx = i % args.num_generations
            print(f"\n[model_output] batch={bidx} sample={group_idx} candidate={cand_idx}")
            print(gen_text)

        # Evaluate candidates.
        group_pass_flags = [False for _ in prompts]
        for i, (prompt, gen_text) in enumerate(zip(repeated_prompts, responses)):
            group_idx = i // args.num_generations
            cand_idx = i % args.num_generations
            meta = metadata_list[group_idx]
            if "timeout_sec" not in meta or "compile_timeout_sec" not in meta:
                meta = dict(meta)
                meta.setdefault("timeout_sec", args.timeout_sec)
                meta.setdefault("compile_timeout_sec", args.compile_timeout_sec)

            candidate_dir = (
                debug_run_dir
                / f"batch_{bidx:04d}"
                / f"sample_{group_idx:04d}"
                / f"candidate_{cand_idx:02d}"
            )
            prev_fixed_dir = os.environ.get("GRPO_TC_FIXED_DIR")
            os.environ["GRPO_TC_FIXED_DIR"] = str(candidate_dir)

            try:
                tc_out = evaluate_test_case(prompt, gen_text, meta, reasoning_mode=args.reasoning_mode)
            finally:
                if prev_fixed_dir is None:
                    os.environ.pop("GRPO_TC_FIXED_DIR", None)
                else:
                    os.environ["GRPO_TC_FIXED_DIR"] = prev_fixed_dir
            passed, tc_score = _parse_test_case_result(tc_out)
            debug_candidate_dir = dump_candidate_debug_artifacts(
                debug_root=debug_run_dir,
                batch_idx=bidx,
                sample_idx=group_idx,
                candidate_idx=cand_idx,
                prompt=prompt,
                generated_text=gen_text,
                metadata=meta,
                tc_out=tc_out if isinstance(tc_out, dict) else {"raw": str(tc_out)},
                passed=bool(passed),
                tc_score=float(tc_score),
            )

            candidate_total += 1
            candidate_pass += int(passed)
            testcase_scores.append(float(tc_score))
            if passed:
                group_pass_flags[group_idx] = True
            else:
                fail_detail_counter[str(tc_out.get("detail", "unknown"))] += 1
                detail = str(tc_out.get("detail", "unknown"))
                compile_cmd = tc_out.get("compile_cmd", "")
                compile_stderr = tc_out.get("compile_stderr", "")
                if detail == "compile_fail" or compile_stderr:
                    print(
                        f"\n[compile_error] batch={bidx} sample={group_idx} candidate={cand_idx} detail={detail}"
                    )
                    print(f"[debug_dir] {debug_candidate_dir}")
                    if compile_cmd:
                        print(f"[compile_cmd] {compile_cmd}")
                    if compile_stderr:
                        print("[compile_stderr]")
                        print(str(compile_stderr))
                    else:
                        print("[compile_stderr] <empty>")

            if args.output_jsonl:
                rec = {
                    "batch": bidx,
                    "sample_index": group_idx,
                    "candidate_index": cand_idx,
                    "passed": bool(passed),
                    "testcase_score": float(tc_score),
                    "detail": tc_out.get("detail", "unknown"),
                    "design_id": meta.get("design_id"),
                    "algo_name": meta.get("algo_name"),
                }
                with open(args.output_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        prompt_total += len(prompts)
        prompt_pass_at_k += sum(1 for x in group_pass_flags if x)

        if bidx % args.log_every == 0:
            cand_rate = candidate_pass / max(candidate_total, 1)
            p_at_k = prompt_pass_at_k / max(prompt_total, 1)
            print(
                f"[batch {bidx}] prompts={prompt_total} "
                f"candidate_pass_rate={cand_rate:.4f} pass@{args.num_generations}={p_at_k:.4f}"
            )

    summary = {
        "model_path": args.model_path,
        "test_data": args.test_data,
        "debug_output_dir": str(debug_run_dir),
        "num_prompts": prompt_total,
        "num_generations": args.num_generations,
        "num_candidates": candidate_total,
        "candidate_pass_rate": candidate_pass / max(candidate_total, 1),
        f"pass@{args.num_generations}": prompt_pass_at_k / max(prompt_total, 1),
        "mean_testcase_score": float(sum(testcase_scores) / max(len(testcase_scores), 1)),
        "fail_details_top20": dict(fail_detail_counter.most_common(20)),
    }

    print("\n" + "=" * 80)
    print("SFT Pass Rate Evaluation")
    print("=" * 80)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Saved summary: {args.output_json}")


if __name__ == "__main__":
    main()
