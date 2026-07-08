"""A single, push-based training-curve plotter shared by every backend.

Both backend adapters call ``.log(epoch, metrics)`` directly from their
own training callbacks, instead of each re-parsing a framework-specific
results CSV after the fact — that CSV parsing (and the header-mismatch
workarounds it needed once a mid-run unfreeze added optimizer param
groups) was the most fragile, duplicated part of the original two
training scripts.
"""

from __future__ import annotations

from pathlib import Path


class TrainingPlotter:
    """Accumulates per-epoch metric dicts and renders multi-panel curves.

    Args:
        out_dir: Directory the PNGs are written to.
        panels: List of ``(title, [(label, metric_key, color), ...])``
            tuples describing how to group metrics into subplots.
        plot_every: Render a snapshot every N epochs. ``.finalize()``
            always renders regardless of this interval.
        title: Figure suptitle prefix.
    """

    def __init__(
        self,
        out_dir: str | Path,
        panels: list[tuple[str, list[tuple[str, str, str]]]],
        plot_every: int = 5,
        title: str = "Training",
    ):
        self.out_dir = Path(out_dir)
        self.panels = panels
        self.plot_every = max(1, plot_every)
        self.title = title
        self.history: list[dict] = []
        self.stage_boundaries: list[int] = []

    def mark_stage_boundary(self, epoch: int) -> None:
        """Record an epoch number where a training stage ended, to draw
        a vertical reference line on future plots."""
        self.stage_boundaries.append(epoch)

    def log(self, epoch: int, metrics: dict[str, float]) -> None:
        """Record one epoch's metrics and render a snapshot if it's due."""
        self.history.append({"epoch": epoch, **metrics})
        if epoch % self.plot_every == 0:
            self._render(epoch, final=False)

    def finalize(self) -> None:
        """Render a final snapshot of the full training history, if any."""
        if self.history:
            self._render(self.history[-1]["epoch"], final=True)

    def _render(self, epoch: int, final: bool) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import pandas as pd

        df = pd.DataFrame(self.history)
        n = len(self.panels)
        cols = min(3, n) or 1
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(6 * cols, 4.5 * rows), squeeze=False
        )
        fig.suptitle(
            f"{self.title} — {'Final' if final else f'Epoch {epoch}'}",
            fontsize=14,
            fontweight="bold",
        )

        for idx, (panel_title, series) in enumerate(self.panels):
            ax = axes[idx // cols][idx % cols]
            has_data = False
            for label, key, color in series:
                if key in df.columns and df[key].notna().any():
                    ax.plot(
                        df["epoch"], df[key], label=label, color=color, linewidth=1.8
                    )
                    has_data = True
            for boundary in self.stage_boundaries:
                if boundary <= epoch:
                    ax.axvline(
                        x=boundary,
                        color="#7f8c8d",
                        linestyle=":",
                        linewidth=1.0,
                        alpha=0.7,
                    )
            ax.set_title(panel_title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Epoch", fontsize=8)
            if has_data:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.25, linestyle="--")

        for idx in range(n, rows * cols):
            axes[idx // cols][idx % cols].axis("off")

        plt.tight_layout()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        tag = "final" if final else f"epoch_{epoch:04d}"
        plt.savefig(
            self.out_dir / f"training_curves_{tag}.png", dpi=130, bbox_inches="tight"
        )
        plt.close(fig)
