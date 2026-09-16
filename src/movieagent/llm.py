"""Thin LLM client wrappers.

Both backends speak the same tiny protocol so `agent.py` does not care which one
is behind it:

    reply = client.complete(system, messages, tools) -> {"text": str, "tool_calls": [...]}

`openai` also covers any OpenAI-compatible endpoint (Groq, Together, DeepSeek,
a local Ollama) through `--base-url`, which is why there is no third backend.

Nothing here is imported unless a key is configured - the default runtime is
fully offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


@dataclass
class LLMReply:
    text: str
    tool_calls: list[dict]          # [{"id": str, "name": str, "arguments": dict}]
    raw: object = None


class OpenAIClient:
    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None, api_key: str | None = None):
        from openai import OpenAI                       # imported lazily, optional dependency

        self.model = model
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"),
                             base_url=base_url or os.environ.get("OPENAI_BASE_URL"))

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        from .tools import openai_tool_schemas

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}] + messages,
            tools=openai_tool_schemas(), temperature=0.3,
        )
        msg = resp.choices[0].message
        calls = [{"id": c.id, "name": c.function.name,
                  "arguments": json.loads(c.function.arguments or "{}")}
                 for c in (msg.tool_calls or [])]
        return LLMReply(text=msg.content or "", tool_calls=calls, raw=msg)


class AnthropicClient:
    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None):
        import anthropic                                # imported lazily, optional dependency

        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMReply:
        resp = self.client.messages.create(
            model=self.model, max_tokens=1600, system=system,
            messages=messages, tools=tools, temperature=0.3,
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "arguments": b.input}
                 for b in resp.content if b.type == "tool_use"]
        return LLMReply(text=text, tool_calls=calls, raw=resp)


def build_client(backend: str, model: str | None = None, base_url: str | None = None):
    if backend == "openai":
        return OpenAIClient(model=model or "gpt-4o-mini", base_url=base_url)
    if backend == "anthropic":
        return AnthropicClient(model=model or "claude-sonnet-5")
    raise ValueError(f"no LLM client for backend {backend!r}")
