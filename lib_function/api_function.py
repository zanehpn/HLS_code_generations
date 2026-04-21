import random
import subprocess
import requests
import re
import os
from pathlib import Path
from typing import List
import shutil
import json
from typing import Dict, List, Union
from openai import OpenAI

SYSTEM_DESCRIBE = (
    "You are an expert in High-Level Synthesis (HLS) and hardware design. "
    "Given C/C++ HLS source code, provide a concise natural language description in English. "
    "Focus on WHAT the function does (inputs, outputs, purpose, behavior), NOT how it is implemented. "
    "Do not describe pragmas, loop structures, or low-level implementation details. "
    "Keep the description brief (2-4 sentences)."
)


def describe_hls_function(file_path: Union[str, Path]) -> str:
    """
    Given an HLS function file path, call the LLM to output a natural language
    English description focusing on functionality (what it does), not implementation details.

    Args:
        file_path: Path to the HLS C/C++ source file

    Returns:
        Natural language English description of the function
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"HLS file not found: {path}")
    code = path.read_text(encoding="utf-8", errors="replace")
    prompt = f"""Describe the functionality of this HLS function in plain English. Focus on what it does, not how it is implemented.

```c
{code}
```

Provide a brief functional description:"""
    return gpt5(prompt, system=SYSTEM_DESCRIBE)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python api_function.py <hls_file_path>")
        sys.exit(1)
    desc = describe_hls_function(sys.argv[1])
    print(desc)


def llm_call(prompt: str, system: str = "", model: str = "gpt-5.1") -> str:
    """
    Call an LLM via OpenAI-compatible API.
    Reads credentials from environment: OPENAI_API_KEY and optional OPENAI_BASE_URL.
    Returns the raw response from the LLM without any processing.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY environment variable before calling llm_call().")
    base_url = os.environ.get("OPENAI_BASE_URL") or None
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        completions = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if completions:
            return completions.choices[0].message.content
    except Exception as e:
        return f"API Error: {str(e)}"


def gpt5(prompt: str, system: str = "") -> str:
    """Backward-compatible wrapper."""
    return llm_call(prompt, system=system, model="gpt-5.1")


