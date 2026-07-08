"""Renders a static report.html: same table as markdown_report.py, plus the PNG charts.

Plain f-strings, no templating engine (jinja2 etc.) — the page is simple
enough that a template dependency wouldn't buy anything.
"""
from __future__ import annotations

from pathlib import Path


def render(rows: list, plot_filenames: list[str], accuracy_keys: list[str] = ()) -> str:
    header_html = "".join(f"<th>{h}</th>" for h in ["Version", "FPS", "Latency", *accuracy_keys, "Size"])
    body_html = ""
    for r in rows:
        acc_cells = "".join(f"<td>{r.accuracy.get(k, 0.0):.1f}</td>" for k in accuracy_keys)
        body_html += (
            f"<tr><td>{r.name}</td><td>{r.fps:.0f}</td><td>{r.latency_ms:.0f} ms</td>"
            f"{acc_cells}<td>{r.size_mb:.0f} MB</td></tr>"
        )
    images_html = "".join(f'<img src="plots/{name}" alt="{name}">' for name in plot_filenames)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Optimization Report</title></head>
<body>
<h1>Optimization Report</h1>
<table border="1"><thead><tr>{header_html}</tr></thead><tbody>{body_html}</tbody></table>
<h2>Plots</h2>
{images_html}
</body></html>"""


def write(rows: list, output_path: Path, plot_filenames: list[str], accuracy_keys: list[str] = ()) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(rows, plot_filenames, accuracy_keys), encoding="utf-8")
    return output_path