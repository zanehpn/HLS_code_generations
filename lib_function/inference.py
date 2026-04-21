"""
NL2HLS inference: generate multiple HLS design candidates from functional descriptions.

Mimics the PPA-RTL flow: Base Model takes Functional Description -> Design¹, Design², ...
"""

import argparse
from pathlib import Path
from typing import List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from .acceleration_utils import VLLMInferenceBackend
from .dataset import SFT_SYSTEM


def _format_generation_prompt(prompt: str) -> str:
    return (
        f"<|im_start|>system\n{SFT_SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def _generate_designs_transformers(
    model_path: str,
    prompts: List[str],
    num_designs: int = 5,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: Optional[str] = None,
) -> List[List[str]]:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    full_prompts = [_format_generation_prompt(prompt) for prompt in prompts]
    inputs = tokenizer(
        full_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_lens = inputs["attention_mask"].sum(dim=1)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            num_return_sequences=num_designs,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    repeated_prompt_lens = prompt_lens.repeat_interleave(num_designs)
    grouped_designs: List[List[str]] = []
    flat_index = 0
    for _prompt in prompts:
        current_group: List[str] = []
        for _ in range(num_designs):
            start = int(repeated_prompt_lens[flat_index].item())
            generated = tokenizer.decode(outputs[flat_index, start:], skip_special_tokens=True)
            if "<|im_end|>" in generated:
                generated = generated.split("<|im_end|>", 1)[0]
            current_group.append(generated.strip())
            flat_index += 1
        grouped_designs.append(current_group)

    return grouped_designs


def _generate_designs_vllm(
    model_path: str,
    prompts: List[str],
    num_designs: int = 5,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
    vllm_dtype: str = "bfloat16",
    vllm_enforce_eager: bool = False,
) -> List[List[str]]:
    backend = VLLMInferenceBackend(
        model_path=model_path,
        tensor_parallel_size=vllm_tensor_parallel_size,
        gpu_memory_utilization=vllm_gpu_memory_utilization,
        max_model_len=vllm_max_model_len,
        dtype=vllm_dtype,
        trust_remote_code=True,
        enforce_eager=vllm_enforce_eager,
    )
    full_prompts = [_format_generation_prompt(prompt) for prompt in prompts]
    return backend.generate(
        prompts=full_prompts,
        num_return_sequences=num_designs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|im_end|>"],
    )


def generate_designs(
    model_path: str,
    prompt: str,
    num_designs: int = 5,
    max_new_tokens: int = 2048,
    temperature: float = 0.8,
    top_p: float = 0.95,
    device: Optional[str] = None,
    backend: str = "transformers",
    vllm_tensor_parallel_size: int = 1,
    vllm_gpu_memory_utilization: float = 0.9,
    vllm_max_model_len: Optional[int] = None,
    vllm_dtype: str = "bfloat16",
    vllm_enforce_eager: bool = False,
) -> List[str]:
    if backend == "vllm":
        return _generate_designs_vllm(
            model_path=model_path,
            prompts=[prompt],
            num_designs=num_designs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            vllm_tensor_parallel_size=vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_max_model_len=vllm_max_model_len,
            vllm_dtype=vllm_dtype,
            vllm_enforce_eager=vllm_enforce_eager,
        )[0]
    return _generate_designs_transformers(
        model_path=model_path,
        prompts=[prompt],
        num_designs=num_designs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        device=device,
    )[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--prompt", type=str, help="Functional description (or from file)")
    parser.add_argument("--prompt_file", type=str, help="Path to file with prompt(s), one per line")
    parser.add_argument("--output_dir", type=str, default="./generated_designs")
    parser.add_argument("--num_designs", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", type=str, default=None, help="Transformers backend device override")
    parser.add_argument("--backend", type=str, choices=["transformers", "vllm"], default="transformers",
                        help="Generation backend")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1,
                        help="vLLM tensor parallel size")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9,
                        help="vLLM GPU memory utilization target")
    parser.add_argument("--vllm_max_model_len", type=int, default=None,
                        help="Optional vLLM max model length override")
    parser.add_argument("--vllm_dtype", type=str, default="bfloat16",
                        help="vLLM dtype (for example: auto, float16, bfloat16)")
    parser.add_argument("--vllm_enforce_eager", action="store_true",
                        help="Force eager mode in vLLM")
    args = parser.parse_args()

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    if args.prompt_file:
        path = Path(args.prompt_file)
        if path.exists():
            prompts.extend([p.strip() for p in path.read_text().splitlines() if p.strip()])

    if not prompts:
        raise ValueError("Provide --prompt or --prompt_file")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_designs} designs per prompt with backend={args.backend}...")
    if args.backend == "vllm":
        all_designs = _generate_designs_vllm(
            model_path=args.model,
            prompts=prompts,
            num_designs=args.num_designs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
            vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
            vllm_max_model_len=args.vllm_max_model_len,
            vllm_dtype=args.vllm_dtype,
            vllm_enforce_eager=args.vllm_enforce_eager,
        )
    else:
        all_designs = _generate_designs_transformers(
            model_path=args.model,
            prompts=prompts,
            num_designs=args.num_designs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

    for i, (prompt, designs) in enumerate(zip(prompts, all_designs)):
        print(f"Saving prompt {i+1}...")
        for j, d in enumerate(designs):
            out_path = out_dir / f"prompt_{i}_design_{j}.cpp"
            out_path.write_text(d, encoding="utf-8")
            print(f"  Saved {out_path}")
        (out_dir / f"prompt_{i}_description.txt").write_text(prompt, encoding="utf-8")


if __name__ == "__main__":
    main()
