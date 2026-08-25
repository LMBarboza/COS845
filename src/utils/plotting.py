"""Charts for Stage 1 and Stage 2 results.

Design rules followed here (they are not cosmetic -- they are what makes a five-series
chart readable):

* One fixed categorical colour per mechanism, assigned by identity and never cycled,
  so a mechanism keeps its colour across every figure in the study.
* Every series carries a direct end-of-line label in addition to the legend, so a
  reader never has to match identity by colour alone.
* One measure per axis; never two y-scales on one chart.
* The ~273 GB/s unified-memory ceiling is drawn as a reference rule on bandwidth
  charts, since the whole thesis is about a bandwidth-bound decode path.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from .metrics import GB10_PEAK_BANDWIDTH_GBPS  # noqa: E402

#: Validated categorical palette, assigned by series identity in fixed order. Eight slots,
#: because a Stage 2 figure carries one series per model and the roster runs to seven --
#: with a five-slot palette the sixth and seventh series silently reused an earlier hue.
SERIES_COLORS: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#dedcd6"

#: Fixed mechanism order, so colours stay stable across figures.
MECHANISM_ORDER = (
    "Naive MHA",
    "FlashAttention (IO-aware)",
    "DeepSeek MLA (reference)",
    "Mamba-2 (SSD)",
    "Kimi Delta Attention",
)


def _color_for(mechanism: str, seen: list[str]) -> str:
    """Stable colour per series: Stage 1 mechanisms hold fixed slots, everything else is
    assigned in first-seen order.

    Categorical hues are never cycled -- two series sharing a colour is a misread figure,
    not a cosmetic problem -- so exhausting the palette raises instead of wrapping. A
    figure that needs a ninth series wants small multiples or an "other" fold, not a
    ninth hue. Stage 1 and Stage 2 series never share an axes, so the two assignment
    schemes cannot collide.
    """
    if mechanism in MECHANISM_ORDER:
        return SERIES_COLORS[MECHANISM_ORDER.index(mechanism)]
    if mechanism not in seen:
        seen.append(mechanism)
    index = seen.index(mechanism)
    if index >= len(SERIES_COLORS):
        raise ValueError(
            f"{index + 1} series exceeds the {len(SERIES_COLORS)}-slot categorical palette; "
            "split the figure into small multiples rather than reusing a hue"
        )
    return SERIES_COLORS[index]


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, pad=12, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def _pow2_formatter() -> FuncFormatter:
    return FuncFormatter(lambda v, _: f"{int(v) // 1024}k" if v >= 1024 else f"{int(v)}")


def load_stage1(path: Path) -> list[dict[str, Any]]:
    """Load the ``results`` array from a Stage 1 JSON output file."""
    return json.loads(path.read_text())["results"]


def load_stage2(path: Path) -> dict[str, Any]:
    """Load a Stage 2 JSON output file whole (metadata plus per-concurrency rows).

    A legend label is derived from the filename stem rather than the ``model`` field: a
    model served from a local directory reports its filesystem path as its id, and two
    runs of the same model at different prompt lengths report an identical id. The stem
    carries both the model and the prompt length, which is what distinguishes the curves.
    """
    payload = json.loads(path.read_text())
    payload.setdefault("label", path.stem.replace("__in", " @ "))
    return payload


def _series(
    rows: Sequence[dict[str, Any]], phase: str, field: str
) -> dict[str, tuple[list[float], list[float]]]:
    """Group ``rows`` into ``{mechanism: (x, y)}`` for one phase and one metric."""
    grouped: dict[str, tuple[list[float], list[float]]] = {}
    for row in rows:
        if row.get("phase") != phase or row.get("status") != "ok":
            continue
        value = row.get(field)
        if value is None:
            continue
        xs, ys = grouped.setdefault(row["mechanism"], ([], []))
        xs.append(float(row["seq_len"]))
        ys.append(float(value))
    for xs, ys in grouped.values():
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs[:], ys[:] = [xs[i] for i in order], [ys[i] for i in order]
    return grouped


#: Minimum vertical separation between direct end-of-line labels, in display pixels.
LABEL_MIN_GAP_PX = 13.0


def _direct_labels(ax: plt.Axes, endpoints: list[tuple[float, float, str]]) -> None:
    """Annotate each series at its last point, nudged apart so labels never overlap.

    Series whose curves converge (naive MHA and the reference MLA sit almost on top of
    each other) would otherwise print their labels in the same place, which defeats the
    point of direct labelling.
    """
    if not endpoints:
        return
    ax.figure.canvas.draw()
    positions = [
        (idx, ax.transData.transform((x, y))[1]) for idx, (x, y, _) in enumerate(endpoints)
    ]
    positions.sort(key=lambda item: item[1])

    adjusted: dict[int, float] = {}
    previous: float | None = None
    for idx, y_px in positions:
        if previous is not None and y_px - previous < LABEL_MIN_GAP_PX:
            y_px = previous + LABEL_MIN_GAP_PX
        adjusted[idx] = y_px
        previous = y_px

    px_to_points = 72.0 / ax.figure.dpi
    for idx, (x, y, text) in enumerate(endpoints):
        original_px = ax.transData.transform((x, y))[1]
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(6, (adjusted[idx] - original_px) * px_to_points),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=7.5,
            va="center",
            annotation_clip=False,
        )


def _plot_lines(
    ax: plt.Axes,
    grouped: dict[str, tuple[list[float], list[float]]],
    log_y: bool = True,
) -> None:
    """Draw one line per mechanism with a direct end-of-line label plus a legend."""
    seen: list[str] = []
    ordered = sorted(
        grouped,
        key=lambda m: MECHANISM_ORDER.index(m) if m in MECHANISM_ORDER else len(MECHANISM_ORDER),
    )
    endpoints: list[tuple[float, float, str]] = []
    for mechanism in ordered:
        xs, ys = grouped[mechanism]
        if not xs:
            continue
        color = _color_for(mechanism, seen)
        ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=4.5, label=mechanism)
        endpoints.append((xs[-1], ys[-1], mechanism))
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.xaxis.set_major_formatter(_pow2_formatter())
    if len(ordered) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    _direct_labels(ax, endpoints)


def _new_figure(width: float = 9.0, height: float = 5.2) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    return fig, ax


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Extra right margin so the direct end-of-line labels are not clipped.
    fig.subplots_adjust(right=0.78)
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_stage1(rows: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Render the Stage 1 figure set. Returns the paths written."""
    written: list[Path] = []

    specs = [
        ("prefill", "latency_ms", "Prefill latency", "Latency (ms)", "prefill_latency.png", True),
        (
            "decode",
            "latency_ms",
            "Decode latency per token",
            "Latency (ms/token)",
            "decode_latency.png",
            True,
        ),
        (
            "prefill",
            "peak_allocated_mb",
            "Peak memory allocated during prefill",
            "Peak allocated (MB)",
            "prefill_peak_memory.png",
            True,
        ),
        (
            "decode",
            "state_memory_mb",
            "Decode state / KV-cache footprint",
            "State memory (MB)",
            "decode_state_memory.png",
            True,
        ),
        (
            "prefill",
            "throughput_tokens_per_s",
            "Effective prefill throughput",
            "Throughput (tokens/s)",
            "prefill_throughput.png",
            True,
        ),
    ]
    for phase, field, title, ylabel, filename, log_y in specs:
        grouped = _series(rows, phase, field)
        if not grouped:
            continue
        fig, ax = _new_figure()
        _plot_lines(ax, grouped, log_y=log_y)
        _style_axes(ax, f"{title} (GB10, batch 1, bfloat16)", "Sequence length S", ylabel)
        written.append(_save(fig, out_dir / filename))

    for phase in ("prefill", "decode"):
        grouped = _series(rows, phase, "bandwidth_gb_per_s")
        if not grouped:
            continue
        fig, ax = _new_figure()
        _plot_lines(ax, grouped, log_y=False)
        ax.axhline(
            GB10_PEAK_BANDWIDTH_GBPS,
            color=TEXT_SECONDARY,
            linewidth=1.2,
            linestyle="--",
            alpha=0.7,
        )
        ax.annotate(
            f"GB10 unified-memory ceiling ~{GB10_PEAK_BANDWIDTH_GBPS:.0f} GB/s",
            xy=(0.02, GB10_PEAK_BANDWIDTH_GBPS),
            xycoords=("axes fraction", "data"),
            xytext=(0, 5),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=8,
        )
        _style_axes(
            ax,
            f"Estimated {phase} memory bandwidth (GB10, batch 1, bfloat16)",
            "Sequence length S",
            "Bandwidth (GB/s)",
        )
        written.append(_save(fig, out_dir / f"{phase}_bandwidth.png"))

    return written


def plot_stage2(payloads: Sequence[dict[str, Any]], out_dir: Path) -> list[Path]:
    """Render the Stage 2 figure set from one or more serving-benchmark payloads."""
    written: list[Path] = []
    specs = [
        ("ttft_mean_ms", "Time to first token", "TTFT (ms)", "ttft.png"),
        ("tpot_mean_ms", "Time per output token", "TPOT (ms/token)", "tpot.png"),
        ("itl_p95_ms", "Inter-token latency, p95", "ITL p95 (ms)", "itl_p95.png"),
        (
            "output_throughput_tokens_per_s",
            "Aggregate output throughput",
            "Throughput (tokens/s)",
            "output_throughput.png",
        ),
        (
            "request_throughput_req_per_s",
            "Request throughput",
            "Throughput (req/s)",
            "request_throughput.png",
        ),
    ]
    for field, title, ylabel, filename in specs:
        fig, ax = _new_figure()
        seen: list[str] = []
        endpoints: list[tuple[float, float, str]] = []
        drawn = 0
        for payload in payloads:
            label = payload.get("label") or payload.get("model", "model")
            rows = payload.get("results", [])
            xs = [float(r["concurrency"]) for r in rows if r.get(field) is not None]
            ys = [float(r[field]) for r in rows if r.get(field) is not None]
            if not xs:
                continue
            color = _color_for(label, seen)
            ax.plot(xs, ys, color=color, linewidth=2.0, marker="o", markersize=4.5, label=label)
            endpoints.append((xs[-1], ys[-1], label))
            drawn += 1
        if not drawn:
            plt.close(fig)
            continue
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        if drawn > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
        _direct_labels(ax, endpoints)
        _style_axes(ax, f"{title} (vLLM on GB10)", "Concurrent clients", ylabel)
        written.append(_save(fig, out_dir / filename))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m src.utils.plotting", description="Render benchmark figures."
    )
    parser.add_argument("--stage1", type=Path, default=None, help="Stage 1 results JSON")
    parser.add_argument(
        "--stage2", type=Path, nargs="*", default=(), help="one or more Stage 2 results JSON files"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    args = parser.parse_args(argv)

    written: list[Path] = []
    if args.stage1:
        written += plot_stage1(load_stage1(args.stage1), args.output_dir / "stage1")
    if args.stage2:
        written += plot_stage2([load_stage2(p) for p in args.stage2], args.output_dir / "stage2")
    if not written:
        parser.error("nothing to plot: pass --stage1 and/or --stage2")
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
