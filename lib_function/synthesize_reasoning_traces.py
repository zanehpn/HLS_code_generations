#!/usr/bin/env python3
import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from api_function import gpt5  # noqa: E402


SYSTEM_PROMPT = """You are an expert HLS engineer creating compact reasoning traces for supervised fine-tuning of a code LLM.

You will receive:
1. A user instruction asking for HLS code.
2. The final HLS code solution.

Write a short reasoning process that could plausibly lead to the final solution.

Requirements:
- The trace is for training a 7B-scale code LLM, so keep it compact and high signal.
- It should read like brief step-by-step reasoning that helps a model decide what code to write next.
- Split the reasoning into exactly two parts:
  1. Functional reasoning
  2. Performance reasoning
- Functional reasoning should sound like: identify the core computation, decide the top-level interface, choose the main state/dataflow, then choose the control structure.
- Performance reasoning should sound like: inspect the latency/resource targets, rule out some pragma choices, then justify the simpler or more aggressive pragma strategy that follows.
- Explicitly connect the requested clock/latency/resource targets to likely implementation choices such as pipeline, unroll, array partition, inline, dataflow, or leaving loops rolled.
- If the final code contains pragmas, explain why those pragmas are reasonable under the given constraints.
- If the final code omits aggressive pragmas, explain why a simpler implementation better matches the stated goals or resource budget.
- Keep the reasoning grounded in the prompt rather than giving generic HLS advice.
- Keep the trace concrete and grounded in the provided task.
- Use 2-4 short bullet points inside each part.
- Each bullet should be one sentence.
- Prefer 8 bullets total or fewer.
- Avoid markdown headings, numbering, bold text, long examples, and tutorial-style explanation.
- Do not restate the full prompt, enumerate all specs, or rewrite the code.
- Avoid noun-heavy spec summaries such as just listing arrays, modules, or constraints without a decision attached.
- Prefer verbs like "need", "choose", "keep", "avoid", "so", and "because" to make the trace read like a decision process.
- Do not mention that you were given the final code.
- Do not include code fences.
- Output exactly in this format:
Functional reasoning:
- ...
Performance reasoning:
- ...
"""


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc


def iter_jsonl_or_empty(path: Path) -> Iterable[Dict]:
    if not path.exists():
        return iter(())
    return iter_jsonl(path)


@dataclass
class PendingEntry:
    input_text: str
    ready_record: Optional[Dict] = None
    future: Optional[Future] = None
    reused: bool = False


def build_reasoning_prompt(task_input: str, final_output: str) -> str:
    return f"""Generate a compact training-time reasoning trace for this HLS code-generation example.

The reasoning should help a code LLM infer:
- what top-level interface and computation structure to choose,
- how the latency / clock / FF / LUT / DSP / BRAM targets should influence pragma choices,
- and why the final implementation style is a reasonable fit for the requested metrics.
- Separate the answer into functional reasoning and performance reasoning.
- Keep the answer compact because it is meant for 7B code-model training.
- Prefer short bullets over detailed explanation.
- Make it sound like reasoning steps, not a design spec or code review.
- In functional reasoning, prefer "I need to model X, so I should choose Y" style decisions.
- In performance reasoning, prefer "Because the target asks for X, I should avoid/use pragma Y" style decisions.
- Do not dump all constraints; mention only the ones that directly change a coding decision.

Task instruction:
{task_input}

Final HLS code:
{final_output}
"""


def normalize_reasoning(reasoning: str) -> str:
    reasoning = reasoning.strip()
    func_match = re.search(
        r"Functional reasoning:\s*(.*?)(?:\n\s*Performance reasoning:|\Z)",
        reasoning,
        flags=re.DOTALL | re.IGNORECASE,
    )
    perf_match = re.search(
        r"Performance reasoning:\s*(.*?)\Z",
        reasoning,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if func_match and perf_match:
        func = func_match.group(1).strip()
        perf = perf_match.group(1).strip()
    else:
        bullets = [ln.strip() for ln in reasoning.splitlines() if ln.strip()]
        if not bullets:
            bullets = ["- Choose a top-level implementation that matches the requested functionality."]
        midpoint = max(1, len(bullets) // 2)
        func = "\n".join(bullets[:midpoint]).strip()
        perf = "\n".join(bullets[midpoint:]).strip()
        if not perf:
            perf = "- Use pragma choices conservatively so the implementation remains aligned with the requested latency and resource targets."
    return (
        "Functional reasoning:\n"
        f"{func}\n\n"
        "Performance reasoning:\n"
        f"{perf}"
    )


def format_augmented_output(reasoning: str, final_output: str, style: str) -> str:
    reasoning = normalize_reasoning(reasoning)
    final_output = final_output.rstrip() + "\n"
    if style == "tagged":
        return (
            "<think>\n"
            f"{reasoning}\n"
            "</think>\n\n"
            "<final_code>\n"
            f"{final_output}"
            "</final_code>\n"
        )
    return (
        "Think:\n"
        f"{reasoning}\n\n"
        "Final Code:\n"
        f"{final_output}"
    )


def has_nonempty_output(record: Dict) -> bool:
    return bool(str(record.get("output", "")).strip())


def synthesize_reasoning(
    task_input: str,
    final_output: str,
    max_retries: int = 0,
    retry_backoff_sec: float = 1.0,
) -> str:
    prompt = build_reasoning_prompt(task_input, final_output)
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = gpt5(prompt, system=SYSTEM_PROMPT).strip()
            if response.startswith("API Error:"):
                raise RuntimeError(response)
            return response
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
            sleep_sec = retry_backoff_sec * (2 ** attempt)
            print(
                f"[warn] retrying after error on attempt {attempt + 1}/{max_retries + 1}: {exc}",
                file=sys.stderr,
            )
            time.sleep(sleep_sec)
    assert last_error is not None
    raise last_error


def process_record(
    record: Dict,
    style: str,
    max_retries: int = 0,
    retry_backoff_sec: float = 1.0,
) -> Dict:
    if "input" not in record or "output" not in record:
        raise KeyError("Each record must contain 'input' and 'output'")

    reasoning = synthesize_reasoning(
        record["input"],
        record["output"],
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    return {
        "input": record["input"],
        "output": format_augmented_output(reasoning, record["output"], style),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment SFT jsonl records with synthetic reasoning traces using api_function.gpt5()."
    )
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parents[2] / "data/sft/designs_with_metrics_random_train_rest_after_unique_test_50.jsonl"),
        help="Input jsonl file with 'input' and 'output' fields.",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).resolve().parents[2] / "data/sft/designs_with_metrics_random_train_rest_after_unique_test_50_with_reasoning.jsonl"),
        help="Output jsonl path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N records.")
    parser.add_argument(
        "--style",
        choices=["tagged", "sections"],
        default="tagged",
        help="How to embed reasoning into the output field.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Compatibility flag; resume behavior is now enabled automatically when the output file exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore any existing output file and regenerate from scratch.",
    )
    parser.add_argument(
        "--skip-failures",
        action="store_true",
        help="Skip records that fail API generation instead of stopping.",
    )
    parser.add_argument(
        "--sleep-sec",
        type=float,
        default=0.0,
        help="Optional sleep between API calls.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of concurrent API requests to issue for missing records.",
    )
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=16,
        help="Maximum number of generated-but-not-yet-written records kept in flight.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Number of retries for a failed API request.",
    )
    parser.add_argument(
        "--retry-backoff-sec",
        type=float,
        default=1.0,
        help="Base backoff in seconds between API retries.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    written = 0
    reused = 0
    skipped = 0

    existing_iter = iter_jsonl_or_empty(output_path) if (output_path.exists() and not args.overwrite) else iter(())
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    reuse_enabled = output_path.exists() and not args.overwrite
    max_workers = max(1, args.num_workers)
    max_inflight = max(max_workers, args.max_inflight)

    pending: Dict[int, PendingEntry] = {}
    next_to_write = 0
    inflight = 0

    def write_entry(out_f, index: int, entry: PendingEntry) -> None:
        nonlocal written, reused, skipped, inflight
        if entry.ready_record is not None:
            out_f.write(json.dumps(entry.ready_record, ensure_ascii=False) + "\n")
            out_f.flush()
            if entry.reused:
                reused += 1
            else:
                written += 1
            return

        assert entry.future is not None
        try:
            record_to_write = entry.future.result()
            out_f.write(json.dumps(record_to_write, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
        except Exception as exc:
            skipped += 1
            print(f"[warn] record {index} failed: {exc}", file=sys.stderr)
            if not args.skip_failures:
                raise
            out_f.write(json.dumps({"input": entry.input_text, "output": ""}, ensure_ascii=False) + "\n")
            out_f.flush()
        finally:
            inflight -= 1

    def flush_ready(out_f, block: bool) -> None:
        nonlocal next_to_write
        while True:
            entry = pending.get(next_to_write)
            if entry is None:
                return
            if entry.ready_record is not None:
                write_entry(out_f, next_to_write, entry)
                del pending[next_to_write]
                next_to_write += 1
                continue
            assert entry.future is not None
            if not entry.future.done() and not block:
                return
            write_entry(out_f, next_to_write, entry)
            del pending[next_to_write]
            next_to_write += 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor, tmp_path.open("w", encoding="utf-8") as out_f:
        for index, record in enumerate(iter_jsonl(input_path)):
            if args.limit is not None and (processed + reused) >= args.limit:
                break

            existing_record = next(existing_iter, None) if reuse_enabled else None
            if existing_record is not None:
                same_input = existing_record.get("input") == record.get("input")
                if same_input and has_nonempty_output(existing_record):
                    pending[index] = PendingEntry(
                        input_text=record["input"],
                        ready_record=existing_record,
                        reused=True,
                    )
                    flush_ready(out_f, block=False)
                    continue
                if not same_input:
                    reuse_enabled = False
                    print(
                        f"[warn] output file diverges from input at record {index}; disabling reuse for remaining records",
                        file=sys.stderr,
                    )

            future = executor.submit(
                process_record,
                record,
                args.style,
                args.max_retries,
                args.retry_backoff_sec,
            )
            pending[index] = PendingEntry(
                input_text=record["input"],
                future=future,
                reused=False,
            )
            inflight += 1
            processed += 1

            flush_ready(out_f, block=False)
            while inflight >= max_inflight:
                flush_ready(out_f, block=True)

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

        while pending:
            flush_ready(out_f, block=True)

    tmp_path.replace(output_path)

    print(
        f"done: generated={written}, reused={reused}, skipped={skipped}, output={output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
