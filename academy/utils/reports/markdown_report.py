"""Renders benchmark rows into report.md, with a naive recommendation line."""
from __future__ import annotations

from pathlib import Path


def recommend(rows: list, accuracy_keys: list[str], max_accuracy_drop_pct: float = 1.0) -> str:
    """Pick the fastest variant that doesn't stray too far from the baseline's accuracy.

    # ponytail: naive heuristic. Assumes rows[0] is the FP32/PyTorch baseline,
    # only looks at accuracy_keys[0], and uses one hard cutoff rather than a
    # weighted score (doesn't factor in size or VRAM). Upgrade path: accept a
    # scoring function instead of a single threshold if this stops being good enough.
    """
    if not rows or not accuracy_keys:
        return "Not enough data to make a recommendation."
    key = accuracy_keys[0]
    baseline = rows[0].accuracy.get(key, 0.0)
    candidates = [r for r in rows if baseline - r.accuracy.get(key, 0.0) <= max_accuracy_drop_pct]
    best = max(candidates or rows, key=lambda r: r.fps)
    return f"\u2714 **{best.name}** offers the best speed within {max_accuracy_drop_pct} pts of baseline {key}."


def render(rows: list, framework: str, checkpoint: str, date: str, accuracy_keys: list[str] = ()) -> str:
    header = ["Version", "FPS", "Latency", *accuracy_keys, "Size"]
    lines = [
        "# Optimization Report",
        "",
        "## Source",
        "",
        f"Framework: {framework}",
        "",
        f"Checkpoint: {checkpoint}",
        "",
        f"Date: {date}",
        "",
        "---",
        "",
        "## Results",
        "",
        f"| {' | '.join(header)} |",
        f"| {' | '.join(['---'] * len(header))} |",
    ]
    for r in rows:
        acc_cells = [f"{r.accuracy.get(k, 0.0):.1f}" for k in accuracy_keys]
        lines.append(f"| {r.name} | {r.fps:.0f} | {r.latency_ms:.0f} ms | {' | '.join(acc_cells)} | {r.size_mb:.0f} MB |")
    lines += ["", "---", "", "## Recommendation", "", recommend(rows, accuracy_keys)]
    return "\n".join(lines)


def write(rows: list, output_path: Path, **render_kwargs) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(rows, **render_kwargs), encoding="utf-8")
    return output_path