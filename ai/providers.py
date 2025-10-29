"""
Provider abstractions for large-language-model (LLM) integrations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional

import requests

from config import ConfigError, settings
from utils.logger import get_logger

LOGGER = get_logger("LLMRegistry")


class LLMProvider(ABC):
    """Minimal interface all LLM providers must satisfy."""

    name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Return a model completion for the supplied prompt."""


class OpenAIProvider(LLMProvider):
    """Simple REST wrapper around OpenAI compatible chat completion endpoint."""

    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_key: Optional[str], model: str) -> None:
        if not api_key:
            raise ConfigError("OPENAI_API_KEY is required to use the OpenAI LLM provider.")
        self.api_key = api_key
        self.model = model
        self.name = "openai"

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = kwargs.get("timeout", 30)
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected response from OpenAI API: {data}") from exc


class DeepSeekProvider(LLMProvider):
    """DeepSeek chat-completion adapter."""

    api_url = "https://api.deepseek.com/v1/chat/completions"

    def __init__(self, api_key: Optional[str], model: str) -> None:
        if not api_key:
            raise ConfigError("DEEPSEEK_API_KEY is required to use the DeepSeek LLM provider.")
        self.api_key = api_key
        self.model = model
        self.name = "deepseek"

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 512),
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        timeout = kwargs.get("timeout", 30)
        response = requests.post(self.api_url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Unexpected response from DeepSeek API: {data}") from exc


_PROVIDERS: Dict[str, LLMProvider] = {}


def register_provider(provider: LLMProvider) -> None:
    """Register a provider instance for later lookup."""
    _PROVIDERS[provider.name] = provider
    LOGGER.debug("Registered LLM provider '%s'", provider.name)


def _build_provider(name: str) -> LLMProvider:
    key = name.lower()
    if key == "deepseek":
        provider = DeepSeekProvider(settings.llm.deepseek_api_key, settings.llm.deepseek_model)
    elif key in {"openai", "default"}:
        provider = OpenAIProvider(settings.llm.openai_api_key, settings.llm.openai_model)
    else:
        raise ConfigError(f"Unsupported LLM provider requested: {name}")
    register_provider(provider)
    return provider


def llm(name: Optional[str] = None) -> LLMProvider:
    """Retrieve an LLM provider by name (defaults to configured provider)."""
    target = (name or settings.llm.default_provider).lower()
    if target not in _PROVIDERS:
        return _build_provider(target)
    return _PROVIDERS[target]
