"""Async OpenAI-compatible streaming client tuned for SGLang, vLLM, MLX, llama.cpp, Ollama, LM Studio.

Streams `reasoning_content` deltas, standard tool-call argument
fragments, and usage (completion_tokens, prompt_tokens, reasoning_tokens).
Auto-detects engine and underlying model quantization metadata.
"""

from __future__ import annotations

import asyncio
import json
import re
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


def extract_quantization_from_text(text: str | None) -> str | None:
    if not text:
        return None
    s = str(text).lower()

    if "nvfp4" in s:
        return "NVFP4"
    if "mxfp4" in s:
        return "MXFP4"
    if re.search(r"\bfp4\b", s) or "-fp4" in s or "_fp4" in s:
        return "FP4"
    if "exl3" in s:
        return "EXL3"
    if "exl2" in s:
        return "EXL2"
    if "fp8" in s or "fp8_e4m3" in s or "fp8_e5m2" in s:
        return "FP8"
    if "awq" in s:
        return "AWQ"
    if "gptq" in s:
        return "GPTQ"

    # Match GGUF quantization patterns (e.g. Q4_K_M, Q8_0, Q5_K_S)
    m_gguf = re.search(r"\b(q\d+_[a-z0-9_]+)\b", s)
    if m_gguf:
        return m_gguf.group(1).upper()

    if "bf16" in s or "bfloat16" in s:
        return "BF16"
    if "fp16" in s or "float16" in s:
        return "FP16"
    if "int4" in s:
        return "INT4"
    if "int8" in s:
        return "INT8"

    return None


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

    async def check(self) -> tuple[list[str], str, str | None]:
        r = await self.client.get("/v1/models")
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        raw_models = data.get("data", [])
        model_ids = [m.get("id") for m in raw_models if m.get("id")]
        engine, detected_quant = await self.detect_engine_and_quant(raw_models)
        return model_ids, engine, detected_quant

    async def detect_engine_and_quant(self, raw_models: list[dict] | None = None) -> tuple[str, str | None]:
        owned_by_set: set[str] = set()
        model_roots: list[str] = []
        model_ids: list[str] = []
        quant_hints: list[str] = []

        if raw_models:
            for m in raw_models:
                if m.get("owned_by"):
                    owned_by_set.add(str(m["owned_by"]).lower())
                if m.get("root"):
                    model_roots.append(str(m["root"]).lower())
                    quant_hints.append(str(m["root"]))
                if m.get("id"):
                    model_ids.append(str(m["id"]).lower())
                    quant_hints.append(str(m["id"]))
                if m.get("quantization"):
                    quant_hints.append(str(m["quantization"]))
                if m.get("format"):
                    quant_hints.append(str(m["format"]))

        has_gguf = any(".gguf" in s or "gguf" in s for s in model_roots + model_ids)
        has_mlx = any("mlx" in s for s in model_roots + model_ids) or "mlx" in owned_by_set

        # Probe server endpoints concurrently for server identity & model metadata
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
            _probe("/get_model_info"),
            _probe("/props"),
            _probe("/slots"),
            _probe("/api/version"),
            _probe("/api/tags"),
            _probe("/info"),
        )
        probe_map = {p: r for p, r in probes if r is not None and r.status_code == 200}

        # Inspect server responses for quantization hints
        for path, resp in probe_map.items():
            try:
                text = resp.text
                quant_hints.append(text)
                if resp.headers.get("content-type", "").startswith("application/json"):
                    j = resp.json()
                    if isinstance(j, dict):
                        for k in ("model_path", "tokenizer_path", "quantization", "load_format", "dtype", "kv_cache_dtype", "model"):
                            if j.get(k):
                                quant_hints.append(str(j[k]))
            except Exception:
                pass

        # Detect quantization from all gathered hints
        detected_quant = None
        for hint in quant_hints:
            q = extract_quantization_from_text(hint)
            if q:
                detected_quant = q
                break

        # Detect engine
        engine = "OpenAI-Compatible"

        if "mlx" in owned_by_set or has_mlx:
            engine = "MLX (mlx-lm)"
        elif "vllm" in owned_by_set:
            engine = "vLLM"
        elif "sglang" in owned_by_set:
            engine = "SGLang"
        elif any(x in owned_by_set for x in ("llamacpp", "llama.cpp", "llama-server", "gguf")):
            engine = "llama.cpp (GGUF)" if has_gguf else "llama.cpp"
        elif "ollama" in owned_by_set:
            engine = "Ollama"
        elif any(x in owned_by_set for x in ("lmstudio", "lm-studio")):
            engine = "LM Studio"
        elif "tgi" in owned_by_set:
            engine = "TGI"
        elif "aphrodite" in owned_by_set:
            engine = "Aphrodite"
        elif any(x in owned_by_set for x in ("tensorrt_llm", "triton")):
            engine = "TensorRT-LLM"
        elif "litellm" in owned_by_set:
            engine = "LiteLLM"
        elif "/" in probe_map and "sglang is running" in probe_map["/"].text.lower():
            engine = "SGLang"
        elif "/get_server_info" in probe_map:
            engine = "SGLang"
        elif "/version" in probe_map:
            engine = "vLLM"
        elif "/" in probe_map and "ollama is running" in probe_map["/"].text.lower():
            engine = "Ollama"
        elif "/api/version" in probe_map or "/api/tags" in probe_map:
            engine = "Ollama"
        elif "/props" in probe_map or "/slots" in probe_map:
            engine = "llama.cpp (GGUF)" if has_gguf else "llama.cpp"
        elif "/" in probe_map and any(x in probe_map["/"].text.lower() for x in ("llama.cpp", "llama-server")):
            engine = "llama.cpp (GGUF)" if has_gguf else "llama.cpp"
        elif "/info" in probe_map:
            engine = "TGI"
        elif has_gguf:
            engine = "llama.cpp (GGUF)"
        elif has_mlx:
            engine = "MLX (mlx-lm)"

        return engine, detected_quant

    async def stream(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        chat_template_kwargs: dict | None = None,
        reasoning_effort: str | None = None,
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
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
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
            rc = delta.get("reasoning_content") or delta.get("reasoning")
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
