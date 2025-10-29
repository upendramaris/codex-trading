"""
Helpers to route DeepSeek/OpenAI prompt templates for the trading agent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from ai import llm
from agent.tools import persist_artifact

PROMPT_DIR = Path("prompts")

AVAILABLE_PROMPTS = {
    "research",
    "features",
    "codegen",
    "critic",
    "risk",
    "ensemble",
    "execution",
    "backtest",
    "gate",
    "runbook",
}


def load_prompt(prompt_name: str) -> str:
    """Return the system prompt text for the requested template."""
    key = prompt_name.lower()
    if key not in AVAILABLE_PROMPTS:
        raise ValueError(f"Unknown prompt '{prompt_name}'. Available: {sorted(AVAILABLE_PROMPTS)}")
    path = PROMPT_DIR / f"{key}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def run_prompt(
    prompt_name: str,
    user_content: str,
    *,
    model: str = "deepseek",
    temperature: float = 0.2,
    max_tokens: int = 1500,
    persist: bool = True,
) -> str:
    """
    Execute a structured prompt against the configured LLM provider.

    Parameters
    ----------
    prompt_name : str
        Name of the prompt template (e.g., "research", "risk").
    user_content : str
        User-specific instructions inserted after the system prompt.
    model : str
        Provider name ("deepseek" or "openai"). Defaults to DeepSeek.
    temperature : float
        Sampling temperature for the provider.
    max_tokens : int
        Maximum tokens in the response (passed through).
    persist : bool
        Persist prompt + response via persist_artifact for audit trail.
    """
    system_prompt = load_prompt(prompt_name)
    provider = llm(model)

    composite_prompt = f"{system_prompt}\n\nUSER\n{user_content.strip()}"
    response = provider.generate(
        composite_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if persist:
        persist_artifact(
            "prompt_trace",
            content={
                "prompt_name": prompt_name,
                "model": provider.name,
                "system_prompt": system_prompt,
                "user_content": user_content,
                "response": response,
            },
        )

    return response.strip()
