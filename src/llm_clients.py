"""Utilities for working with Azure-hosted language models.

The UI supports a catalogue of Azure OpenAI and Azure AI Inference models.
This module exposes light-weight helper classes so the rest of the
application can remain framework agnostic.  The helpers deliberately avoid
depending on the Azure SDK to keep the example self-contained; instead they
issue HTTP requests directly against the REST endpoints that both services
expose.

Environment variables control endpoint discovery and authentication:

```
AZURE_OPENAI_ENDPOINT       Base URL for Azure OpenAI deployments
AZURE_OPENAI_API_KEY        API key for Azure OpenAI
AZURE_OPENAI_API_VERSION    Optional API version (defaults to 2024-02-15-preview)

AZURE_AI_ENDPOINT           Base URL for Azure AI Inference deployments
AZURE_AI_API_KEY            API key for Azure AI Inference
AZURE_AI_API_VERSION        Optional API version (defaults to 2024-05-01-preview)
```

The helpers raise descriptive exceptions instead of returning bare strings so
callers can decide how to inform the user.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

import requests


class LLMConfigurationError(RuntimeError):
    """Raised when an LLM call cannot be prepared due to misconfiguration."""


class LLMCapabilityError(RuntimeError):
    """Raised when a model does not support the requested capability."""


@dataclass(frozen=True)
class ProviderConfig:
    """Low level provider metadata used to construct API calls."""

    endpoint_env: str
    api_key_env: str
    api_version_env: str
    default_api_version: str
    path_template: str

    def resolve_endpoint(self) -> str:
        endpoint = os.getenv(self.endpoint_env)
        if not endpoint:
            raise LLMConfigurationError(
                f"Set the {self.endpoint_env} environment variable to call this model."
            )
        return endpoint.rstrip("/")

    def resolve_key(self) -> str:
        key = os.getenv(self.api_key_env)
        if not key:
            raise LLMConfigurationError(
                f"Set the {self.api_key_env} environment variable to call this model."
            )
        return key

    def resolve_api_version(self) -> str:
        return os.getenv(self.api_version_env, self.default_api_version)


PROVIDERS: Mapping[str, ProviderConfig] = {
    "azure": ProviderConfig(
        endpoint_env="AZURE_OPENAI_ENDPOINT",
        api_key_env="AZURE_OPENAI_API_KEY",
        api_version_env="AZURE_OPENAI_API_VERSION",
        default_api_version="2024-02-15-preview",
        path_template="/openai/deployments/{deployment}/chat/completions",
    ),
    "azure_ai": ProviderConfig(
        endpoint_env="AZURE_AI_ENDPOINT",
        api_key_env="AZURE_AI_API_KEY",
        api_version_env="AZURE_AI_API_VERSION",
        default_api_version="2024-05-01-preview",
        path_template="/openai/deployments/{deployment}/chat/completions",
    ),
}


@dataclass(frozen=True)
class LLMModelConfig:
    """Describes a model entry in the UI catalogue."""

    model_id: str
    provider: str
    capability: str = "chat"
    description: Optional[str] = None

    def provider_config(self) -> ProviderConfig:
        try:
            return PROVIDERS[self.provider]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise LLMConfigurationError(
                f"Unsupported provider '{self.provider}' for model {self.model_id}"
            ) from exc


MODEL_CATALOGUE: List[LLMModelConfig] = [
    LLMModelConfig("azure/genailab-maas-gpt-35-turbo", provider="azure"),
    LLMModelConfig("azure/genailab-maas-gpt-4o", provider="azure"),
    LLMModelConfig("azure/genailab-maas-gpt-4o-mini", provider="azure"),
    LLMModelConfig(
        "azure/genailab-maas-text-embedding-3-large",
        provider="azure",
        capability="embedding",
        description="Embeddings model – not suitable for direct chat responses.",
    ),
    LLMModelConfig(
        "azure/genailab-maas-whisper",
        provider="azure",
        capability="audio",
        description="Whisper transcription model – not suitable for chat responses.",
    ),
    LLMModelConfig("azure_ai/genailab-maas-DeepSeek-R1", provider="azure_ai"),
    LLMModelConfig("azure_ai/genailab-maas-DeepSeek-V3-0324", provider="azure_ai"),
    LLMModelConfig(
        "azure_ai/genailab-maas-Llama-3.2-90B-Vision-Instruct",
        provider="azure_ai",
        capability="multimodal",
        description="Vision-capable instruct model.",
    ),
    LLMModelConfig("azure_ai/genailab-maas-Llama-3.3-70B-Instruct", provider="azure_ai"),
    LLMModelConfig(
        "azure_ai/genailab-maas-Llama-4-Maverick-17B-128E-Instruct-FP8",
        provider="azure_ai",
    ),
    LLMModelConfig(
        "azure_ai/genailab-maas-Phi-3.5-vision-instruct",
        provider="azure_ai",
        capability="multimodal",
    ),
    LLMModelConfig("azure_ai/genailab-maas-Phi-4-reasonin", provider="azure_ai"),
]


class AzureChatClient:
    """Minimal HTTP client for Azure-hosted chat completion models."""

    def __init__(self, model: LLMModelConfig) -> None:
        self.model = model
        self.provider = model.provider_config()

    def _build_url(self) -> str:
        endpoint = self.provider.resolve_endpoint()
        path = self.provider.path_template.format(deployment=self.model.model_id)
        api_version = self.provider.resolve_api_version()
        return f"{endpoint}{path}?api-version={api_version}"

    def generate(self, messages: Iterable[Mapping[str, str]], **kwargs: object) -> str:
        if self.model.capability != "chat":
            raise LLMCapabilityError(
                f"Model {self.model.model_id} does not support chat completions."
            )

        url = self._build_url()
        headers = {
            "Content-Type": "application/json",
            "api-key": self.provider.resolve_key(),
        }
        payload: MutableMapping[str, object] = {
            "messages": list(messages),
            "temperature": kwargs.get("temperature", 0.3),
            "top_p": kwargs.get("top_p", 0.95),
        }
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            raise RuntimeError(
                f"Model request failed ({response.status_code}): {response.text}"
            )

        data: Dict[str, object] = response.json()
        choices = data.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned by the model response")
        first = choices[0]
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str):
            raise RuntimeError("Malformed response payload from the model")
        return content


def find_model_config(model_id: str) -> LLMModelConfig:
    """Lookup helper that raises a descriptive error if the model is unknown."""

    for model in MODEL_CATALOGUE:
        if model.model_id == model_id:
            return model
    raise LLMConfigurationError(f"Unknown model '{model_id}'")


__all__ = [
    "AzureChatClient",
    "LLMModelConfig",
    "LLMCapabilityError",
    "LLMConfigurationError",
    "MODEL_CATALOGUE",
    "find_model_config",
]

