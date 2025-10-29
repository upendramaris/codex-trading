"""
Lightweight LLM registry and convenience accessors.
"""

from .providers import (
    DeepSeekProvider,
    LLMProvider,
    OpenAIProvider,
    llm,
    register_provider,
)

__all__ = [
    "LLMProvider",
    "OpenAIProvider",
    "DeepSeekProvider",
    "llm",
    "register_provider",
]
