from __future__ import annotations

import json
from pathlib import Path


def _fmt(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}f}"


def _safe_get(d: dict, path: list[str], default=None):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_report(metrics_path: Path, output_html: Path) -> None:
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)

    baseline_psnr = float(_safe_get(metrics, ["baseline", "psnr"], 0.0))
    baseline_mse = float(_safe_get(metrics, ["baseline", "mse"], 0.0))
    hq_psnr = float(_safe_get(metrics, ["hq", "psnr"], 0.0))
    hq_mse = float(_safe_get(metrics, ["hq", "mse"], 0.0))

    psnr_gain = hq_psnr - baseline_psnr
    mse_drop = baseline_mse - hq_mse

    baseline_cfg = _safe_get(metrics, ["baseline", "config"], {})
    hq_cfg = _safe_get(metrics, ["hq", "config"], {})

    grid_rows = metrics.get("grid", [])
    grid_rows = sorted(grid_rows, key=lambda r: float(r.get("psnr", 0.0)), reverse=True)

    best_grid = grid_rows[0] if grid_rows else None

    images_root = Path("images/output/part1_neural_field")
    paths = {
        "baseline_final": (images_root / "baseline_final.png").as_posix(),
        "hq_final": (images_root / "hq_final.png").as_posix(),
        "baseline_prog": (images_root / "baseline_progression.png").as_posix(),
        "hq_prog": (images_root / "hq_progression.png").as_posix(),
        "baseline_psnr": (images_root / "baseline_psnr_curve.png").as_posix(),
        "hq_psnr": (images_root / "hq_psnr_curve.png").as_posix(),
        "grid": (images_root / "grid_2x2_results.png").as_posix(),
    }

    grid_table_rows = ""
    for row in grid_rows:
        grid_table_rows += (
            "<tr>"
            f"<td>{int(row.get('L', 0))}</td>"
            f"<td>{int(row.get('width', 0))}</td>"
            f"<td>{_fmt(float(row.get('psnr', 0.0)), 2)}</td>"
            f"<td>{_fmt(float(row.get('mse', 0.0)), 6)}</td>"
            "</tr>"
        )

    best_grid_text = "N/A"
    if best_grid is not None:
        best_grid_text = (
            f"L={int(best_grid.get('L', 0))}, width={int(best_grid.get('width', 0))}, "
            f"PSNR={_fmt(float(best_grid.get('psnr', 0.0)), 2)} dB"
        )

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Part 1 Diff Report</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --fg: #1d2430;
      --muted: #596579;
      --card: #ffffff;
      --line: #d4dce8;
      --up: #0f7b5f;
      --down: #9e2e2e;
      --accent: #204a87;
    }}
    body {{ margin: 0; font-family: Segoe UI, Tahoma, sans-serif; background: var(--bg); color: var(--fg); }}
    .wrap {{ max-width: 1120px; margin: 20px auto 40px; padding: 0 14px; }}
    .hero, .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 14px; }}
    .hero {{ box-shadow: 0 6px 20px rgba(20, 30, 50, 0.07); }}
    h1, h2, h3 {{ margin: 0 0 10px; }}
    h2 {{ margin-top: 22px; }}
    p, li {{ color: var(--muted); line-height: 1.5; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; margin-top: 10px; }}
    .kpi {{ font-size: 28px; font-weight: 700; color: var(--accent); }}
    .up {{ color: var(--up); font-weight: 700; }}
    .down {{ color: var(--down); font-weight: 700; }}
    .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
    img {{ width: 100%; border-radius: 8px; border: 1px solid #e2e7ef; }}
    table {{ width: 100%; border-collapse: collapse; margin-top: 8px; }}
    th, td {{ border-bottom: 1px solid var(--line); text-align: left; padding: 8px 10px; font-size: 14px; }}
    th {{ background: #eef2f8; }}
    .mono {{ font-family: Consolas, monospace; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <section class=\"hero\">
      <h1>Part 1 Results Diff Report</h1>
      <p>
        Auto-generated report comparing baseline and high-quality runs, plus full deliverables
        (progressions, PSNR curves, and grid search).
      </p>
      <p class=\"mono\">Source metrics: {metrics_path.as_posix()}</p>
    </section>

    <h2>Baseline vs HQ Diff</h2>
    <div class=\"kpi-grid\">
      <div class=\"card\"><div>Baseline PSNR</div><div class=\"kpi\">{_fmt(baseline_psnr)} dB</div><div class=\"mono\">MSE {_fmt(baseline_mse, 6)}</div></div>
      <div class=\"card\"><div>HQ PSNR</div><div class=\"kpi\">{_fmt(hq_psnr)} dB</div><div class=\"mono\">MSE {_fmt(hq_mse, 6)}</div></div>
      <div class=\"card\"><div>PSNR Improvement</div><div class=\"kpi up\">+{_fmt(psnr_gain)} dB</div><div>MSE Change: <span class=\"down\">-{_fmt(mse_drop, 6)}</span></div></div>
    </div>

    <h3>Config Diff</h3>
    <table>
      <tr><th>Field</th><th>Baseline</th><th>HQ</th></tr>
      <tr><td>L</td><td>{baseline_cfg.get('L', 'N/A')}</td><td>{hq_cfg.get('L', 'N/A')}</td></tr>
      <tr><td>Width</td><td>{baseline_cfg.get('width', 'N/A')}</td><td>{hq_cfg.get('width', 'N/A')}</td></tr>
      <tr><td>Hidden layers</td><td>{baseline_cfg.get('hidden_layers', 'N/A')}</td><td>{hq_cfg.get('hidden_layers', 'N/A')}</td></tr>
      <tr><td>LR</td><td>{baseline_cfg.get('lr', 'N/A')}</td><td>{hq_cfg.get('lr', 'N/A')}</td></tr>
      <tr><td>Min LR</td><td>{baseline_cfg.get('min_lr', 'N/A')}</td><td>{hq_cfg.get('min_lr', 'N/A')}</td></tr>
      <tr><td>Steps</td><td>{baseline_cfg.get('steps', 'N/A')}</td><td>{hq_cfg.get('steps', 'N/A')}</td></tr>
      <tr><td>Batch size</td><td>{baseline_cfg.get('batch_size', 'N/A')}</td><td>{hq_cfg.get('batch_size', 'N/A')}</td></tr>
    </table>

    <h2>Final Reconstructions</h2>
    <div class=\"grid2\">
      <div class=\"card\"><h3>Baseline Final</h3><img src=\"{paths['baseline_final']}\" alt=\"Baseline final\"></div>
      <div class=\"card\"><h3>HQ Final</h3><img src=\"{paths['hq_final']}\" alt=\"HQ final\"></div>
    </div>

    <h2>Training Progressions</h2>
    <div class=\"card\"><h3>Baseline Progression</h3><img src=\"{paths['baseline_prog']}\" alt=\"Baseline progression\"></div>
    <div class=\"card\"><h3>HQ Progression</h3><img src=\"{paths['hq_prog']}\" alt=\"HQ progression\"></div>

    <h2>PSNR Curves</h2>
    <div class=\"grid2\">
      <div class=\"card\"><h3>Baseline PSNR Curve</h3><img src=\"{paths['baseline_psnr']}\" alt=\"Baseline PSNR curve\"></div>
      <div class=\"card\"><h3>HQ PSNR Curve</h3><img src=\"{paths['hq_psnr']}\" alt=\"HQ PSNR curve\"></div>
    </div>

    <h2>2x2 Hyperparameter Grid</h2>
    <div class=\"card\"><img src=\"{paths['grid']}\" alt=\"Grid 2x2 results\"></div>
    <p><strong>Best grid config:</strong> {best_grid_text}</p>
    <table>
      <tr><th>L</th><th>Width</th><th>PSNR (dB)</th><th>MSE</th></tr>
      {grid_table_rows}
    </table>

    <h2>Deliverables Coverage</h2>
    <ul>
      <li>Model architecture details: covered in config diff table.</li>
      <li>Training progression on provided image: baseline + HQ progression panels.</li>
      <li>Final 2x2 results for two L and two width values: grid figure + table.</li>
      <li>PSNR curve for selected run: baseline/HQ PSNR curves.</li>
    </ul>
  </div>
</body>
</html>
"""

    output_html.write_text(html, encoding="utf-8")


def main() -> None:
    project_root = Path(__file__).resolve().parent
    metrics_path = project_root / "images" / "output" / "part1_neural_field" / "report_metrics.json"
    output_html = project_root / "part1_results_diff_report.html"

    build_report(metrics_path=metrics_path, output_html=output_html)
    print(f"Generated report: {output_html}")


if __name__ == "__main__":
    main()
