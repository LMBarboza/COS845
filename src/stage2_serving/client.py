"""Async streaming client for an OpenAI-compatible vLLM endpoint.

Stage 2 measures what a client actually experiences, so timing here is client-side
wall clock (``time.perf_counter``) around HTTP stream events -- the CUDA-event rule
governs Stage 1 kernel measurement, which is a different thing. Every inter-token
interval is recorded individually so the sweep can report tail latency, not just means.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import aiohttp

#: Sentinel that terminates an OpenAI-compatible SSE stream.
STREAM_DONE = "[DONE]"


@dataclass
class StreamResult:
    """Timings for a single streamed completion request."""

    success: bool
    ttft_ms: float | None = None
    total_ms: float | None = None
    itl_ms: list[float] = field(default_factory=list)
    output_tokens: int = 0
    prompt_tokens: int | None = None
    error: str | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    @property
    def tpot_ms(self) -> float | None:
        """(Total time - TTFT) / (output tokens - 1), in ms per output token."""
        if not self.success or self.ttft_ms is None or self.total_ms is None:
            return None
        if self.output_tokens < 2:
            return None
        return (self.total_ms - self.ttft_ms) / (self.output_tokens - 1)


def _extract_text(chunk: dict) -> str:
    """Pull generated text out of either a completions or chat-completions chunk."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    if "text" in choice:
        return choice.get("text") or ""
    delta = choice.get("delta") or {}
    return delta.get("content") or ""


async def stream_completion(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float = 0.0,
    ignore_eos: bool = True,
    timeout_s: float = 600.0,
) -> StreamResult:
    """Issue one streaming completion and record TTFT plus every inter-token interval.

    ``ignore_eos`` is a vLLM extension: it forces exactly ``max_tokens`` output tokens
    so that every request in a concurrency sweep does the same amount of decode work.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if ignore_eos:
        payload["ignore_eos"] = True

    result = StreamResult(success=False)
    url = f"{endpoint.rstrip('/')}/completions"
    start = time.perf_counter()
    result.started_at = start
    last_token_at = start

    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=timeout_s)
        ) as response:
            if response.status != 200:
                body = (await response.text())[:400]
                result.error = f"HTTP {response.status}: {body}"
                return result

            async for raw in response.content:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == STREAM_DONE:
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if usage := chunk.get("usage"):
                    result.prompt_tokens = usage.get("prompt_tokens")

                if not _extract_text(chunk):
                    continue

                now = time.perf_counter()
                if result.ttft_ms is None:
                    result.ttft_ms = (now - start) * 1000.0
                else:
                    result.itl_ms.append((now - last_token_at) * 1000.0)
                last_token_at = now
                result.output_tokens += 1
    except (TimeoutError, aiohttp.ClientError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        return result

    end = time.perf_counter()
    result.finished_at = end
    result.total_ms = (end - start) * 1000.0
    if result.ttft_ms is None:
        result.error = "stream produced no tokens"
        return result
    result.success = True
    return result


async def discover_model(session: aiohttp.ClientSession, endpoint: str) -> str:
    """Return the id of the first model the endpoint serves."""
    url = f"{endpoint.rstrip('/')}/models"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
        response.raise_for_status()
        body = await response.json()
    models = body.get("data") or []
    if not models:
        raise RuntimeError(f"{url} returned no models")
    return models[0]["id"]


async def scrape_metrics(session: aiohttp.ClientSession, base_url: str) -> dict[str, float]:
    """Scrape the vLLM Prometheus endpoint into a flat ``{metric_name: value}`` map.

    Labels are dropped: the sweep runs one model per server, so the label set carries
    no information the result rows do not already hold. Returns ``{}`` if the endpoint
    is unreachable, since ``/metrics`` is a convenience, not a required data source.
    """
    url = f"{base_url.rstrip('/')}/metrics"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status != 200:
                return {}
            text = await response.text()
    except (TimeoutError, aiohttp.ClientError):
        return {}

    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        name, _, value = line.partition(" ")
        name = name.split("{", 1)[0]
        try:
            metrics[name] = float(value)
        except ValueError:
            continue
    return metrics
