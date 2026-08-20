"""Async OpenAI-compatible streaming client tuned for SGLang + reasoning models.

SGLang streams `reasoning_content` deltas, standard tool-call argument
fragments, and a final chunk (with empty `choices`) carrying exact
`usage.completion_tokens` / `usage.reasoning_tokens`.

Nemo/Nemotron reasoning models accept `chat_template_kwargs` (e.g.
`{"enable_thinking": bool}`) to toggle the reasoning block.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx


@dataclass
class ToolCall:
    id: str = ""
    name: str = ""
    arguments: str = ""


@dataclass
class StreamResult:
    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    error: str | None = None
    elapsed_s: float = 0.0
    ttft_s: float | None = None


class ChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 600.0, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
            headers=headers,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def check(self) -> tuple[list[str], str]:
        r = await self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        raw_models = data.get("data", [])
        model_ids = [m.get("id") for m in raw_models if m.get("id")]
        engine = await self.detect_engine(raw_models)
        return model_ids, engine

    async def detect_engine(self, raw_models: list[dict] | None = None) -> str:
        owned_by_set: set[str] = set()
        model_roots: list[str] = []
        model_ids: list[str] = []
        if raw_models:
            for m in raw_models:
                if m.get("owned_by"):
                    owned_by_set.add(str(m["owned_by"]).lower())
                if m.get("root"):
                    model_roots.append(str(m["root"]).lower())
                if m.get("id"):
                    model_ids.append(str(m["id"]).lower())

        has_gguf = any(".gguf" in s or "gguf" in s for s in model_roots + model_ids)
        has_mlx = any("mlx" in s for s in model_roots + model_ids) or "mlx" in owned_by_set

        if "mlx" in owned_by_set or has_mlx:
            return "MLX (mlx-lm)"
        if "vllm" in owned_by_set:
            return "vLLM"

        if "sglang" in owned_by_set:
            return "SGLang"

        if any(x in owned_by_set for x in ("llamacpp", "llama.cpp", "llama-server", "gguf")):
            return "llama.cpp (GGUF)" if has_gguf else "llama.cpp"

        if "ollama" in owned_by_set:
            return "Ollama"

        if any(x in owned_by_set for x in ("lmstudio", "lm-studio")):
            return "LM Studio"

        if "tgi" in owned_by_set:
            return "TGI"

        if "aphrodite" in owned_by_set:
            return "Aphrodite"

        if any(x in owned_by_set for x in ("tensorrt_llm", "triton")):
            return "TensorRT-LLM"

        if "litellm" in owned_by_set:
            return "LiteLLM"

        # Probe server endpoints concurrently for server identity
        async def _probe(path: str) -> tuple[str, httpx.Response | None]:
            try:
                res = await self.client.get(path)
                return path, res
            except Exception:
                return path, None

        probes = await asyncio.gather(
            _probe("/"),
            _probe("/version"),
            _probe("/get_server_info"),
            _probe("/props"),
            _probe("/slots"),
            _probe("/api/version"),
            _probe("/info"),
        )
        probe_map = {p: r for p, r in probes if r is not None and r.status_code == 200}

        # Check SGLang
        if "/" in probe_map and "sglang is running" in probe_map["/"].text.lower():
            return "SGLang"
        if "/get_server_info" in probe_map:
            return "SGLang"

        # Check vLLM
        if "/version" in probe_map:
            return "vLLM"

        # Check Ollama
        if "/" in probe_map and "ollama is running" in probe_map["/"].text.lower():
            return "Ollama"
        if "/api/version" in probe_map:
            return "Ollama"

        # Check llama.cpp / GGUF
        if "/props" in probe_map or "/slots" in probe_map:
            return "llama.cpp (GGUF)" if has_gguf else "llama.cpp"
        if "/" in probe_map and any(x in probe_map["/"].text.lower() for x in ("llama.cpp", "llama-server")):
            return "llama.cpp (GGUF)" if has_gguf else "llama.cpp"

        # Check TGI
        if "/info" in probe_map:
            return "TGI"

        if has_gguf:
            return "llama.cpp (GGUF)"

        if has_mlx:
            return "MLX (mlx-lm)"
        return "OpenAI-Compatible"

    async def stream(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        chat_template_kwargs: dict | None = None,
        on_chunk: Callable[[dict], None] | None = None,
    ) -> StreamResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream_options": {"include_usage": True},
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if tools:
            payload["tools"] = tools

        result = StreamResult()
        tc_by_index: dict[int, ToolCall] = {}
        try:
            async with self.client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode("utf-8", "replace")
                    result.error = f"HTTP {resp.status_code}: {body[:400]}"
                    return result
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:"):].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    self._apply(chunk, result, tc_by_index)
                    if on_chunk is not None:
                        on_chunk(chunk)
        except httpx.HTTPError as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        result.tool_calls = [tc_by_index[i] for i in sorted(tc_by_index)]
        return result

    @staticmethod
    def _apply(chunk: dict, result: StreamResult, tc_by_index: dict[int, ToolCall]) -> None:
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta") or {}
            rc = delta.get("reasoning_content")
            ct = delta.get("content")
            if rc:
                result.reasoning += rc
            if ct:
                result.content += ct
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                entry = tc_by_index.setdefault(idx, ToolCall())
                if tc.get("id"):
                    entry.id = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    entry.name += fn["name"]
                if fn.get("arguments"):
                    entry.arguments += fn["arguments"]
            fr = choices[0].get("finish_reason")
            if fr:
                result.finish_reason = fr
        usage = chunk.get("usage")
        if usage:
            result.prompt_tokens = usage.get("prompt_tokens") or 0
            result.completion_tokens = usage.get("completion_tokens") or 0
            # SGLang: usage.reasoning_tokens — vLLM/OpenAI: usage.completion_tokens_details.reasoning_tokens
            reas = usage.get("reasoning_tokens") or 0
            if not reas:
                details = usage.get("completion_tokens_details") or {}
                reas = details.get("reasoning_tokens") or 0
            result.reasoning_tokens = reas
