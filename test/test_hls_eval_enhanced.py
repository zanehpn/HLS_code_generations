"""
Enhanced HLS-EVAL test:
1. Include header content (types + function signatures) in the prompt
2. Add -I _tb_compat to compilation for HLS compat headers
"""
import argparse
import json
import os
import re
import resource
import signal
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.test_local_models import (
    find_all_benchmarks, extract_code,
    HLS_EVAL_DATA_DIR, SYSTEM_HLS_GEN,
)

TB_COMPAT_DIR = Path(__file__).resolve().parents[2] / "forgehls/kernels/kernels/_tb_compat"
COMPILE_TIMEOUT = 60
RUN_TIMEOUT = 120

FIXED_43_BENCHMARKS = [
    'c2hlsc/block', 'c2hlsc/mix_columns', 'c2hlsc/shift_rows', 'c2hlsc/sub_bytes',
    'chstone/df_add128', 'chstone/df_countLeadingZeros32', 'chstone/df_countLeadingZeros64',
    'chstone/df_extractFloat64Exp', 'chstone/df_extractFloat64Frac', 'chstone/df_extractFloat64Sign',
    'chstone/df_float64_abs', 'chstone/df_float64_ge', 'chstone/df_float64_is_nan',
    'chstone/df_float64_is_signaling_nan', 'chstone/df_float64_le', 'chstone/df_float64_neg',
    'chstone/df_mul64To128', 'chstone/df_packFloat64', 'chstone/df_propagateFloat64NaN',
    'chstone/df_shift64RightJamming', 'chstone/dfadd', 'chstone/dfdiv', 'chstone/dfmul', 'chstone/dfsin',
    'flowgnn/fgnn_linear', 'flowgnn/fgnn_linear_input_stationary', 'flowgnn/fgnn_linear_output_stationary',
    'gnnbuilder/compute_neighbor_tables', 'gnnbuilder/gather_node_neighbors', 'gnnbuilder/global_add_pool',
    'machsuite/aes_aes', 'machsuite/bfs_bulk', 'machsuite/stencil_stencil2d',
    'polybench/gemver', 'polybench/seidel-2d',
    'pp4fpga/parallel_merge_sort', 'pp4fpga/pp4fpga_cordic',
    'rosetta/rendering_3d__check_clockwise', 'rosetta/rendering_3d__clockwise_vertices',
    'rosetta/rendering_3d__pixel_in_triangle', 'rosetta/rendering_3d__projection',
    'rosetta/spam_filter__computeGradient', 'rosetta/spam_filter__dotProduct',
]


def build_enhanced_prompt(bench: Dict[str, Any]) -> str:
    """Build prompt with header content (types + function signatures) included."""
    desc = bench["description"].strip()
    top_fn = bench["top_function"]

    # Read header files for type definitions and function signatures
    header_content = ""
    for h in bench.get("h_files", []):
        try:
            h_text = h.read_text(encoding="utf-8", errors="replace").strip()
            if h_text:
                header_content += f"\n\nHeader file `{h.name}`:\n```c\n{h_text}\n```"
        except Exception:
            pass

    # Read the kernel file to extract the function signature
    kernel_sig = ""
    try:
        kernel_code = bench["kernel_file"].read_text(encoding="utf-8", errors="replace")
        # Find the function signature
        pattern = rf"(?:(?:static|inline|void|int|float|double|bool|unsigned|long|short|char|uint\w+|int\w+|float\w+|bits\w+|flag|struct\s+\w+|[\w:]+\s*\*?)\s+)+{re.escape(top_fn)}\s*\([^)]*\)"
        m = re.search(pattern, kernel_code, re.DOTALL)
        if m:
            kernel_sig = m.group(0).strip()
    except Exception:
        pass

    parts = [desc]

    if header_content:
        parts.append(f"\nThe following header file defines the types and function signatures used by this kernel:{header_content}")

    if kernel_sig:
        parts.append(f"\nThe function signature is:\n```c\n{kernel_sig}\n```")

    parts.append(f"\nThe top-level function name must be `{top_fn}`.")
    parts.append(f"\nYou must `#include` the provided header file. Return only the complete C/C++ source code.")

    return "\n".join(parts)


MEM_LIMIT_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB per subprocess


def _set_limits():
    """Pre-exec: new process group + memory limit for child processes."""
    os.setpgrp()
    try:
        resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    except (ValueError, resource.error):
        pass


def _run_with_kill(cmd, timeout, cwd=None, preexec_fn=None):
    """Run a command; on timeout or error, kill the entire process group."""
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=cwd, preexec_fn=preexec_fn,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        # Kill entire process group
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        proc.wait()
        return None, "", ""


def compile_and_run_enhanced(bench, candidate_code):
    """Compile with -I _tb_compat for HLS compat headers.
    Kills hung/OOM subprocesses via process groups + RLIMIT_AS."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="hls_eval_enh_"))
    try:
        # Copy all benchmark files
        for f in bench["dir"].iterdir():
            if f.is_file():
                shutil.copy2(f, tmp_dir / f.name)

        # Write candidate code
        kernel_name = bench["kernel_file"].name
        (tmp_dir / kernel_name).write_text(candidate_code, encoding="utf-8")

        tb_name = bench["tb_file"].name
        tb_path = tmp_dir / tb_name
        exe_path = tmp_dir / "candidate.out"

        if tb_path.suffix == ".cpp" or bench["kernel_file"].suffix == ".cpp":
            cmd = ["g++", "-std=gnu++17", "-O2", "-fpermissive", "-w"]
        else:
            cmd = ["gcc", "-std=gnu11", "-O2", "-w"]

        # Add include paths: tmp_dir (for local headers) + _tb_compat (for HLS compat)
        cmd.extend(["-I", str(tmp_dir)])
        if TB_COMPAT_DIR.exists():
            cmd.extend(["-I", str(TB_COMPAT_DIR)])

        cmd.extend([str(tb_path), str(tmp_dir / kernel_name)])
        cmd.extend(["-lm", "-lstdc++", "-o", str(exe_path)])

        rc, stdout, stderr = _run_with_kill(cmd, timeout=COMPILE_TIMEOUT, preexec_fn=_set_limits)
        if rc is None:
            return {"pass": False, "compiled": False, "detail": "compile_timeout"}
        if rc != 0:
            return {"pass": False, "compiled": False, "detail": "compile_fail",
                    "compile_stderr": stderr[-500:]}

        # Run testbench with memory limit + process group kill on timeout
        run_cmd = [str(exe_path)]
        rc, stdout, stderr = _run_with_kill(
            run_cmd, timeout=RUN_TIMEOUT, cwd=str(tmp_dir), preexec_fn=_set_limits,
        )
        if rc is None:
            return {"pass": False, "compiled": True, "detail": "run_timeout"}

        # Check for OOM (killed by signal or alloc failure)
        if rc == -9 or rc == 137:
            return {"pass": False, "compiled": True, "detail": "run_oom",
                    "stdout_tail": stdout[-1200:] if stdout else "",
                    "stderr_tail": stderr[-1200:] if stderr else ""}

        out_stripped = stdout.strip()
        last_line = out_stripped.splitlines()[-1].strip().lower() if out_stripped.splitlines() else ""
        passed = rc == 0 and any(kw in last_line for kw in ["pass", "success", "correct"])
        if not passed and rc == 0 and "fail" not in last_line and "error" not in last_line:
            passed = True

        result = {"pass": passed, "compiled": True,
                  "detail": "pass" if passed else "functional_fail"}
        if not passed:
            result["stdout_tail"] = stdout[-1200:] if stdout else ""
            result["stderr_tail"] = stderr[-1200:] if stderr else ""
            result["run_exit_code"] = rc
        return result
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num_generations", type=int, default=1)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.45)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    all_bench = find_all_benchmarks(HLS_EVAL_DATA_DIR)
    bench_map = {b["key"]: b for b in all_bench}
    benchmarks = [bench_map[k] for k in FIXED_43_BENCHMARKS if k in bench_map]
    print(f"Fixed 43 benchmarks: found {len(benchmarks)}/{len(FIXED_43_BENCHMARKS)}")

    model_short = args.model.split("/")[-1]
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{model_short}_enhanced_{run_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
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
    sampling_params = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, n=args.num_generations)

    # Build enhanced prompts
    prompts = [build_enhanced_prompt(b) for b in benchmarks]

    if is_chat:
        chat_msgs = [[{"role": "system", "content": SYSTEM_HLS_GEN}, {"role": "user", "content": p}] for p in prompts]
        outputs = llm.chat(chat_msgs, sampling_params=sampling_params)
    else:
        formatted = [f"{SYSTEM_HLS_GEN}\n\n{p}" for p in prompts]
        outputs = llm.generate(formatted, sampling_params=sampling_params)

    total = 0
    compile_pass = 0
    func_pass = 0
    fail_counter = Counter()
    per_bench = []

    for idx, (bench, output) in enumerate(zip(benchmarks, outputs)):
        for gen_idx, gen in enumerate(output.outputs):
            code = extract_code(gen.text)
            result = compile_and_run_enhanced(bench, code)
            total += 1
            compiled = result.get("compiled", False)
            passed = result.get("pass", False)
            compile_pass += int(compiled)
            func_pass += int(passed)
            if not passed:
                fail_counter[result.get("detail", "unknown")] += 1

            # Save debug
            dbg = output_dir / f"{idx:04d}_{bench['key'].replace('/', '_')}" / f"gen_{gen_idx}"
            dbg.mkdir(parents=True, exist_ok=True)
            (dbg / "prompt.txt").write_text(prompts[idx], encoding="utf-8")
            (dbg / "response.txt").write_text(gen.text or "", encoding="utf-8")
            (dbg / "code.txt").write_text(code or "", encoding="utf-8")
            with open(dbg / "result.json", "w") as f:
                json.dump(result, f, indent=2)

        bench_passed = any(output.outputs[i] and compile_and_run_enhanced(bench, extract_code(output.outputs[0].text)).get("pass") for i in [0])
        per_bench.append({"benchmark": bench["key"], "passed": passed, "compiled": compiled, "detail": result.get("detail", "")})

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(benchmarks)}] synth={compile_pass}/{total} func={func_pass}/{total}")

    synth_rate = compile_pass / max(total, 1)
    func_rate = func_pass / max(total, 1)

    summary = {
        "model": args.model,
        "prompt_format": "enhanced (header + signature + tb_compat)",
        "num_benchmarks": len(benchmarks),
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "synthesizability": synth_rate,
        "functional_correctness": func_rate,
        "compile_pass": compile_pass,
        "func_pass": func_pass,
        "total": total,
        "fail_details": dict(fail_counter.most_common(20)),
        "per_benchmark": per_bench,
    }

    print(f"\n{'='*60}")
    print(f"HLS-EVAL 43 Enhanced Results: {model_short}")
    print(f"{'='*60}")
    print(f"Synth = {synth_rate:.4f} ({compile_pass}/{total})")
    print(f"Func  = {func_rate:.4f} ({func_pass}/{total})")

    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
