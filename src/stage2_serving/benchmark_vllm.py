"""Stage 2: production serving benchmark against a running vLLM server.

Sweeps client concurrency against a live OpenAI-compatible endpoint and records the
serving metrics the study calls for: TTFT, TPOT, inter-token latency (mean, standard
deviation, p95), aggregate output throughput, request throughput, and the KV/state
cache footprint scraped from vLLM's Prometheus endpoint.

The server is *not* started here. Launch it separately (see ``scripts/run_stage2.sh``),
then point this client at it -- one process per model, so that KV-cache and state
memory readings belong unambiguously to one architecture.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp

from .client import StreamResult, discover_model, scrape_metrics, stream_completion

#: vLLM Prometheus gauges that report cache occupancy, newest name first.
CACHE_USAGE_METRICS = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc")


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _stdev(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


@dataclass
class ConcurrencyResult:
    """Aggregate serving metrics at one concurrency level."""

    concurrency: int
    requests_completed: int
    requests_failed: int
    input_tokens: int
    output_tokens_requested: int
    output_tokens_generated: int
    wall_time_s: float

    ttft_mean_ms: float
    ttft_median_ms: float
    ttft_p95_ms: float
    tpot_mean_ms: float
    tpot_p95_ms: float
    itl_mean_ms: float
    itl_std_ms: float
    itl_p95_ms: float

    output_throughput_tokens_per_s: float
    request_throughput_req_per_s: float

    kv_cache_usage_perc: float | None = None
    kv_cache_usage_peak_perc: float | None = None
    kv_cache_gb: float | None = None
    kv_cache_peak_gb: float | None = None
    server_metrics: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def flat_row(self) -> dict[str, Any]:
        """CSV-friendly view: nested server metrics dropped, errors collapsed."""
        row = self.as_dict()
        row.pop("server_metrics", None)
        row["errors"] = "; ".join(self.errors[:3])
        return row


def summarize(
    concurrency: int,
    results: Sequence[StreamResult],
    wall_time_s: float,
    input_tokens: int,
    output_tokens: int,
) -> ConcurrencyResult:
    """Reduce a batch of completed streams into one aggregate record."""
    ok = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms is not None]
    itls = [v for r in ok for v in r.itl_ms]
    generated = sum(r.output_tokens for r in ok)

    return ConcurrencyResult(
        concurrency=concurrency,
        requests_completed=len(ok),
        requests_failed=len(failed),
        input_tokens=input_tokens,
        output_tokens_requested=output_tokens,
        output_tokens_generated=generated,
        wall_time_s=wall_time_s,
        ttft_mean_ms=_mean(ttfts),
        ttft_median_ms=statistics.median(ttfts) if ttfts else float("nan"),
        ttft_p95_ms=_percentile(ttfts, 0.95),
        tpot_mean_ms=_mean(tpots),
        tpot_p95_ms=_percentile(tpots, 0.95),
        itl_mean_ms=_mean(itls),
        itl_std_ms=_stdev(itls),
        itl_p95_ms=_percentile(itls, 0.95),
        output_throughput_tokens_per_s=generated / wall_time_s if wall_time_s > 0 else float("nan"),
        request_throughput_req_per_s=len(ok) / wall_time_s if wall_time_s > 0 else float("nan"),
        errors=[r.error for r in failed if r.error],
    )


def build_prompt(input_tokens: int, tokenizer_name: str | None) -> tuple[str, int]:
    """Build a prompt of (approximately) ``input_tokens`` tokens.

    With a tokenizer the length is exact, which matters because TTFT scales with real
    prompt length. Without one, fall back to a repeated single word and report the
    word count so the approximation is visible in the results rather than implied.
    """
    if tokenizer_name:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
            base = tok.encode(
                "the quick brown fox jumps over the lazy dog ", add_special_tokens=False
            )
            if base:
                ids = (base * (input_tokens // len(base) + 1))[:input_tokens]
                return tok.decode(ids), len(ids)
        except Exception as exc:  # noqa: BLE001 -- fall back rather than abort the sweep
            print(
                f"warning: tokenizer {tokenizer_name!r} unusable ({exc}); using word-count prompt"
            )
    return ("word " * input_tokens).strip(), input_tokens


async def sample_peak_cache_usage(
    session: aiohttp.ClientSession,
    metrics_base: str,
    stop: asyncio.Event,
    interval_s: float = 0.25,
) -> float | None:
    """Poll ``/metrics`` until ``stop`` is set and return the highest cache usage seen.

    The cache occupancy gauge falls back to zero once a level's requests drain, so a
    single scrape taken after the level finishes reports an empty cache rather than the
    footprint the load actually required. Sampling during the run is the only way to
    capture the metric the study asks for.
    """
    peak: float | None = None
    while not stop.is_set():
        metrics = await scrape_metrics(session, metrics_base)
        for key in CACHE_USAGE_METRICS:
            if key in metrics:
                value = metrics[key]
                peak = value if peak is None else max(peak, value)
                break
        try:
            await asyncio.wait_for(stop.wait(), timeout=interval_s)
        except TimeoutError:
            continue
    return peak


async def run_concurrency_level(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    concurrency: int,
    requests: int,
    output_tokens: int,
) -> tuple[list[StreamResult], float]:
    """Fire ``requests`` completions with at most ``concurrency`` in flight."""
    semaphore = asyncio.Semaphore(concurrency)

    async def one() -> StreamResult:
        async with semaphore:
            return await stream_completion(session, endpoint, model, prompt, output_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*(one() for _ in range(requests)))
    return list(results), time.perf_counter() - start


async def run_sweep(args: argparse.Namespace) -> int:
    concurrencies = [int(c) for c in args.concurrency.split(",") if c.strip()]
    metrics_base = args.metrics_url or args.endpoint.rstrip("/").removesuffix("/v1")

    async with aiohttp.ClientSession() as session:
        model = args.model
        if not model:
            try:
                model = await discover_model(session, args.endpoint)
            except Exception as exc:  # noqa: BLE001
                print(f"error: could not reach {args.endpoint}: {exc}", file=sys.stderr)
                return 1
        print(f"Serving benchmark against {args.endpoint} (model: {model})", flush=True)

        prompt, prompt_tokens = build_prompt(args.input_tokens, args.tokenizer or model)
        print(
            f"Prompt: {prompt_tokens} tokens; generating {args.output_tokens} tokens/request",
            flush=True,
        )

        if args.warmup_requests:
            print(f"Warming up with {args.warmup_requests} request(s)...", flush=True)
            await run_concurrency_level(
                session,
                args.endpoint,
                model,
                prompt,
                min(args.warmup_requests, 4),
                args.warmup_requests,
                min(args.output_tokens, 16),
            )

        records: list[ConcurrencyResult] = []
        for concurrency in concurrencies:
            requests = args.requests_per_client * concurrency
            print(f"\n-- concurrency={concurrency} ({requests} requests)", flush=True)
            stop = asyncio.Event()
            sampler = asyncio.create_task(sample_peak_cache_usage(session, metrics_base, stop))
            try:
                results, wall = await run_concurrency_level(
                    session, args.endpoint, model, prompt, concurrency, requests, args.output_tokens
                )
            finally:
                stop.set()
            peak_usage = await sampler
            record = summarize(concurrency, results, wall, prompt_tokens, args.output_tokens)

            server_metrics = await scrape_metrics(session, metrics_base)
            record.server_metrics = server_metrics
            for key in CACHE_USAGE_METRICS:
                if key in server_metrics:
                    record.kv_cache_usage_perc = server_metrics[key]
                    if args.kv_cache_size_gb:
                        record.kv_cache_gb = server_metrics[key] * args.kv_cache_size_gb
                    break
            record.kv_cache_usage_peak_perc = peak_usage
            if peak_usage is not None and args.kv_cache_size_gb:
                record.kv_cache_peak_gb = peak_usage * args.kv_cache_size_gb
            records.append(record)
            _print_record(record)

            if args.settle_seconds:
                await asyncio.sleep(args.settle_seconds)

    _write_outputs(args, model, prompt_tokens, records)
    return 0


def _print_record(rec: ConcurrencyResult) -> None:
    print(
        f"   TTFT  mean={rec.ttft_mean_ms:8.1f} ms  p95={rec.ttft_p95_ms:8.1f} ms\n"
        f"   TPOT  mean={rec.tpot_mean_ms:8.2f} ms/token\n"
        f"   ITL   mean={rec.itl_mean_ms:8.2f} ms  std={rec.itl_std_ms:7.2f}  "
        f"p95={rec.itl_p95_ms:8.2f} ms\n"
        f"   out   {rec.output_throughput_tokens_per_s:9.1f} tok/s   "
        f"req {rec.request_throughput_req_per_s:6.2f} req/s   "
        f"failed={rec.requests_failed}",
        flush=True,
    )
    if rec.kv_cache_usage_peak_perc is not None:
        gb = f" ({rec.kv_cache_peak_gb:.2f} GB)" if rec.kv_cache_peak_gb is not None else ""
        print(f"   cache peak {rec.kv_cache_usage_peak_perc * 100:.1f}%{gb}", flush=True)


def _write_outputs(
    args: argparse.Namespace,
    model: str,
    prompt_tokens: int,
    records: Sequence[ConcurrencyResult],
) -> None:
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or model.replace("/", "_")

    payload = {
        "finished_at": datetime.now(UTC).isoformat(),
        "endpoint": args.endpoint,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "output_tokens": args.output_tokens,
        "requests_per_client": args.requests_per_client,
        "results": [r.as_dict() for r in records],
    }
    json_path = out_dir / f"{tag}.json"
    json_path.write_text(json.dumps(payload, indent=2))

    csv_path = out_dir / f"{tag}.csv"
    rows = [r.flat_row() for r in records]
    if rows:
        with csv_path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"\nWrote {json_path}\nWrote {csv_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.stage2_serving.benchmark_vllm",
        description="Stage 2 vLLM serving benchmark (concurrency sweep).",
    )
    parser.add_argument("--endpoint", default="http://localhost:8000/v1")
    parser.add_argument(
        "--model", default=None, help="model id; discovered from /v1/models if omitted"
    )
    parser.add_argument("--tokenizer", default=None, help="tokenizer for exact prompt sizing")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32,64")
    parser.add_argument("--input-tokens", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument(
        "--requests-per-client",
        type=int,
        default=4,
        help="requests issued per concurrent client at each level (default: 4)",
    )
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=2.0,
        help="idle pause between concurrency levels so queues and caches drain",
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="base URL of the vLLM Prometheus endpoint (default: endpoint without /v1)",
    )
    parser.add_argument(
        "--kv-cache-size-gb",
        type=float,
        default=None,
        help="total KV/state cache size in GB, from the server startup log; converts "
        "the reported usage fraction into an absolute footprint",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/raw/stage2"))
    parser.add_argument("--tag", default=None, help="output filename stem (default: model id)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(run_sweep(args))


if __name__ == "__main__":
    raise SystemExit(main())
