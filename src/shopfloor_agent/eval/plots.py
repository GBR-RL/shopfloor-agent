"""Charts for the results, each rendered for a light and a dark page (README <picture>).

Palette: four categorical slots validated for colour-vision deficiency in both modes; aqua and
yellow sit below 3:1 on the light surface, so every chart carries value labels or comes with a
table. Bars are thin with 4 px rounded ends and a 2 px surface gap; grid lines are hairlines.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

THEMES: dict[str, dict[str, Any]] = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]},
    "dark": {"surface": "#1a1a19", "text": "#ffffff", "muted": "#c3c2b7", "grid": "#33332f",
             "series": ["#3987e5", "#d95926", "#199e70", "#c98500"]},
}  # fmt: skip
TIER_LABELS = {
    "overall": "Overall",
    "lookup": "Lookup",
    "aggregate": "Aggregate",
    "multistep": "Multi-step",
    "action": "Action",
    "unanswerable": "Unanswerable",
}


def _axes(theme: dict[str, Any], size: tuple[float, float]) -> tuple[Any, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size, dpi=160)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def _rounded_bar(ax: Any, x: float, width: float, height: float, color: str) -> None:
    """A bar with a rounded data end and a square baseline."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    if height <= 0:  # a zero is data: show it as a thin mark on the baseline, not as nothing
        ax.add_patch(Rectangle((x, 0), width, 0.008, linewidth=0, facecolor=color))
        return
    r = min(0.012, height / 2)  # rounding in data units of the 0..1 axis
    ax.add_patch(FancyBboxPatch((x, 0), width, height, boxstyle=f"round,pad=0,rounding_size={r}",
                                mutation_aspect=1 / 3, linewidth=0, facecolor=color))  # fmt: skip
    ax.add_patch(Rectangle((x, 0), width, min(height, r * 3), linewidth=0, facecolor=color))


def _title(fig: Any, theme: dict[str, Any], title: str, subtitle: str) -> None:
    fig.text(0.02, 0.97, title, ha="left", va="top", fontsize=12.5, fontweight="bold",
             color=theme["text"])  # fmt: skip
    fig.text(0.02, 0.905, subtitle, ha="left", va="top", fontsize=9, color=theme["muted"])


def _save(fig: Any, out: Path, mode: str) -> Path:
    import matplotlib.pyplot as plt

    path = out.with_name(f"{out.name}-{mode}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def tier_chart(
    runs: dict[str, dict[str, Any]], out: Path, *, title: str, subtitle: str
) -> list[Path]:
    """Pass rate per tier (and overall) for up to four runs, with 95 % interval whiskers."""
    if len(runs) > 4:
        raise ValueError("at most four runs per chart (validated palette slots)")
    groups = ["overall", *[t for t in TIER_LABELS if t != "overall"]]
    written = []
    for mode, theme in THEMES.items():
        fig, ax = _axes(theme, (9.2, 4.4))
        n = len(runs)
        gap = 0.02  # the surface gap between touching bars
        width = min(0.12, (0.8 - gap * (n - 1)) / n)  # thin bars: ~24 px at this size
        band = n * width + (n - 1) * gap
        for i, (name, summary) in enumerate(runs.items()):
            color = theme["series"][i]
            for g, group in enumerate(groups):
                block = summary["overall"] if group == "overall" else summary["tiers"].get(group)
                if not block or math.isnan(block["pass_rate"]):
                    continue
                x = g - band / 2 + i * (width + gap)
                _rounded_bar(ax, x, width, block["pass_rate"], color)
                lo, hi = block["ci95"]
                cx = x + width / 2
                ax.plot([cx, cx], [lo, hi], color=theme["muted"], linewidth=1, alpha=0.8)
            overall = summary["overall"]["pass_rate"]
            # the overall score rides in the legend: labels over thin bars would collide
            ax.bar([0], [0], color=color, label=f"{name}: {100 * overall:.0f}% overall")
        ax.set_xlim(-0.6, len(groups) - 0.4)
        ax.set_ylim(0, 1.08)
        ax.set_xticks(range(len(groups)), [TIER_LABELS[g] for g in groups])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        legend = ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=n,
                           bbox_to_anchor=(0.5, -0.09), handlelength=1, handleheight=1)  # fmt: skip
        for text in legend.get_texts():
            text.set_color(theme["text"])
        _title(fig, theme, title, subtitle)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.2)
        written.append(_save(fig, out, mode))
    return written


def cost_chart(
    points: dict[str, tuple[float, float]], out: Path, *, title: str, subtitle: str
) -> list[Path]:
    """Pass rate against median seconds per task; one labelled dot per run."""
    written = []
    for mode, theme in THEMES.items():
        fig, ax = _axes(theme, (7.2, 4.2))
        ax.grid(axis="x", color=theme["grid"], linewidth=0.8)
        for name, (seconds, rate) in points.items():
            ax.scatter([seconds], [rate], s=70, color=theme["series"][0], zorder=3,
                       edgecolors=theme["surface"], linewidths=2)  # fmt: skip
            ax.annotate(name, (seconds, rate), xytext=(7, 4), textcoords="offset points",
                        fontsize=8.5, color=theme["text"])  # fmt: skip
        ax.set_xlabel("median seconds per task (4 vCPU runner)", color=theme["muted"],
                      fontsize=9)  # fmt: skip
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlim(left=0, right=max(s for s, _ in points.values()) * 1.35)
        _title(fig, theme, title, subtitle)
        fig.subplots_adjust(left=0.1, right=0.97, top=0.8, bottom=0.14)
        written.append(_save(fig, out, mode))
    return written
