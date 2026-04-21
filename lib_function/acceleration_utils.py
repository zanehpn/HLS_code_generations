"""
Shared acceleration helpers for training and inference.
"""

from __future__ import annotations

import importlib.util
from typing import Any


def ensure_optional_dependency(module_name: str, use_case: str, pip_name: str | None = None) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    package_name = pip_name or module_name
    raise ImportError(
        f"{module_name} is required for {use_case}. Install it first, for example: `pip install {package_name}`."
    )


def normalize_deepspeed_device(device: str | None) -> str | None:
    value = (device or "none").strip().lower()
    if value in {"", "none", "null"}:
        return None
    if value not in {"cpu", "nvme"}:
        raise ValueError(f"Unsupported DeepSpeed offload device: {device}")
    return value


def build_deepspeed_config(
    train_micro_batch_size: int,
    gradient_accumulation_steps: int,
    zero_stage: int = 2,
    bf16: bool = True,
    gradient_clipping: float | None = None,
    offload_optimizer_device: str | None = None,
    offload_param_device: str | None = None,
    zero3_save_16bit_model: bool = True,
) -> dict[str, Any]:
    if zero_stage not in {0, 1, 2, 3}:
        raise ValueError(f"Unsupported DeepSpeed ZeRO stage: {zero_stage}")

    optimizer_device = normalize_deepspeed_device(offload_optimizer_device)
    param_device = normalize_deepspeed_device(offload_param_device)

    zero_optimization: dict[str, Any] = {"stage": int(zero_stage)}
    if zero_stage >= 1:
        zero_optimization.update(
            {
                "overlap_comm": True,
                "contiguous_gradients": True,
            }
        )
    if zero_stage >= 2:
        zero_optimization["reduce_bucket_size"] = "auto"
    if optimizer_device is not None:
        zero_optimization["offload_optimizer"] = {
            "device": optimizer_device,
            "pin_memory": optimizer_device == "cpu",
        }
    if zero_stage == 3:
        zero_optimization.update(
            {
                "stage3_prefetch_bucket_size": "auto",
                "stage3_param_persistence_threshold": "auto",
                "stage3_gather_16bit_weights_on_model_save": bool(zero3_save_16bit_model),
            }
        )
        if param_device is not None:
            zero_optimization["offload_param"] = {
                "device": param_device,
                "pin_memory": param_device == "cpu",
            }

    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": int(train_micro_batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "zero_optimization": zero_optimization,
        "steps_per_print": 2000,
    }
    if gradient_clipping is not None:
        config["gradient_clipping"] = float(gradient_clipping)

    if bf16:
        config["bf16"] = {"enabled": True}
        config["fp16"] = {"enabled": False}
    else:
        config["bf16"] = {"enabled": False}
        config["fp16"] = {"enabled": True}

    return config


class VLLMInferenceBackend:
    def __init__(
        self,
        model_path: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int | None = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        enforce_eager: bool = False,
    ):
        ensure_optional_dependency("vllm", "vLLM inference", pip_name="vllm")
        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": int(tensor_parallel_size),
            "gpu_memory_utilization": float(gpu_memory_utilization),
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len is not None and int(max_model_len) > 0:
            llm_kwargs["max_model_len"] = int(max_model_len)
        if enforce_eager:
            llm_kwargs["enforce_eager"] = True

        self._llm = LLM(**llm_kwargs)
        self._sampling_params_cls = SamplingParams

    def generate(
        self,
        prompts: list[str],
        num_return_sequences: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        sampling_params = self._sampling_params_cls(
            n=int(num_return_sequences),
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stop=stop or None,
        )
        outputs = self._llm.generate(prompts, sampling_params, use_tqdm=False)

        grouped: list[list[str]] = []
        for request_output in outputs:
            choices = sorted(request_output.outputs, key=lambda output: output.index)
            grouped.append([choice.text.strip() for choice in choices])
        return grouped
