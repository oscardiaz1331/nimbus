"""One PNG bar chart per metric, comparing variants side by side.

Rows are duck-typed (only attribute access, no import of benchmark.py's
dataclass) so reports/ doesn't need a hard dependency on benchmark/.
"""
from __future__ import annotations

from pathlib import Path

_METRICS = {
    # attribute name on the row -> (chart title, y-axis label)
    "latency_ms": ("Latency", "ms"),
    "fps": ("Throughput", "FPS"),
    "size_mb": ("Model size", "MB"),
    "ram_mb": ("RAM usage", "MB"),
}


def plot_metric(names: list[str], values: list[float], title: str, ylabel: str, output_path: Path) -> Path:
    import matplotlib
    matplotlib.use("Agg")  # headless: training boxes usually have no display
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(names, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def generate_all(rows: list, plots_dir: Path, accuracy_keys: list[str] = ()) -> dict[str, Path]:
    """Generate the standard chart set, plus one chart per accuracy key present on the rows."""
    names = [r.name for r in rows]
    generated = {}
    for attr, (title, ylabel) in _METRICS.items():
        values = [getattr(r, attr) for r in rows]
        generated[attr] = plot_metric(names, values, title, ylabel, plots_dir / f"{attr}.png")
    for key in accuracy_keys:
        values = [r.accuracy.get(key, 0.0) for r in rows]
        generated[key] = plot_metric(names, values, key, key, plots_dir / f"{key}.png")
    return generated