"""Generate README SVG charts from saved result CSV files.

These charts are portfolio visualizations rebuilt from project outputs in
`results/`. They are not exported screenshots from the original presentation.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
RESULTS = ROOT / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def bar_svg(
    path: Path,
    rows: list[dict[str, str]],
    label_key: str,
    value_key: str,
    title: str,
    color: str,
    width: int = 880,
    row_h: int = 34,
) -> None:
    items = [(row[label_key], float(row[value_key])) for row in rows]
    max_val = max(value for _, value in items) or 1
    left = 250
    top = 64
    height = top + len(items) * row_h + 48

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="24" y="34" font-family="Arial, sans-serif" font-size="24" font-weight="700" fill="#1f2933">{title}</text>',
    ]

    for index, (label, value) in enumerate(items):
        y = top + index * row_h
        bar_w = int((width - left - 110) * value / max_val)
        display_label = f"{label[:34]}..." if len(label) > 37 else label
        lines.append(f'<text x="24" y="{y + 21}" font-family="Arial, sans-serif" font-size="15" fill="#263238">{display_label}</text>')
        lines.append(f'<rect x="{left}" y="{y + 5}" width="{bar_w}" height="20" rx="3" fill="{color}"/>')
        lines.append(f'<text x="{left + bar_w + 8}" y="{y + 21}" font-family="Arial, sans-serif" font-size="14" fill="#374151">{value:.3f}</text>')

    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)

    scores = read_csv(RESULTS / "clustering_model_scores.csv")
    bar_svg(
        ASSETS / "clustering-silhouette.svg",
        scores,
        "Method",
        "Silhouette",
        "Clustering Silhouette Score",
        "#2E86AB",
    )

    label_map = {
        "acl_pft_rt_z_sor": "Cumulative return score",
        "trk_err_z_sor": "Tracking error score",
        "shpr_z_sor": "Sharpe score",
        "dividend_num": "Dividend frequency",
        "dividend_total": "Dividend amount",
        "yr1_tot_pft_rt": "1Y total return",
        "ifo_rt_z_sor": "Information ratio score",
        "mm3_tot_pft_rt": "3M total return",
        "vty_z_sor": "Volatility score",
        "crr_z_sor": "Correlation score",
        "mxdd_z_sor": "Max drawdown score",
        "mm1_tot_pft_rt": "1M total return",
    }
    features = sorted(
        read_csv(RESULTS / "cluster_feature_importance.csv"),
        key=lambda row: float(row["importance"]),
        reverse=True,
    )[:10]
    for row in features:
        row["label"] = label_map.get(row["feature"], row["feature"])
    bar_svg(
        ASSETS / "cluster-feature-importance.svg",
        features,
        "label",
        "importance",
        "Top Cluster Feature Importance",
        "#33658A",
    )

    keywords = sorted(
        read_csv(RESULTS / "keyword_importance_shap.csv"),
        key=lambda row: float(row["importance"]),
        reverse=True,
    )[:12]
    bar_svg(
        ASSETS / "keyword-importance.svg",
        keywords,
        "feature",
        "importance",
        "Top SHAP Keywords for Return Prediction",
        "#C74C3C",
    )


if __name__ == "__main__":
    main()
