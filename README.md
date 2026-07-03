# Gen Pick: ETF Clustering and Generative AI Summary

Gen Pick is a finance data-product prototype from the 2024 NH Investment & Securities Big Data Competition. It groups ETFs by measurable investment, risk, dividend, and customer-holding indicators, selects representative ETFs for each cluster, and uses generative AI plus SHAP keyword evidence to make the ETF composition easier to inspect.

This repository is a public-safe portfolio version. It keeps the explainable analysis path, selected result tables, presentation assets, and code structure, while excluding contest-restricted raw data and private NH materials.

## Quick Review Path

1. Start with this README for the problem, role, data boundary, evidence, and limitations.
2. Read [docs/methodology.md](docs/methodology.md) for the clustering, recommendation, generative AI summary, and SHAP workflow.
3. Read [docs/data-notice.md](docs/data-notice.md) before reusing any data or asset.
4. Inspect [results/](results/) for public-safe outputs, especially `clustering_model_scores.csv`, `cluster_feature_importance.csv`, `cluster_metric_means.csv`, and `keyword_importance_shap.csv`.
5. Review the presentation-derived evidence images in [assets/](assets/) and the pipeline code in [src/](src/).

## Project Snapshot

| Item | Detail |
| --- | --- |
| Period | 2024.08 - 2024.10 |
| Context | 2024 NH Investment & Securities Big Data Competition |
| Domain | ETF recommendation, clustering, explainability, finance data product |
| Public deliverable | ETF type clustering, representative ETF selection, generative AI ETF summaries, SHAP keyword evidence |
| Stack | Python, pandas, scikit-learn, XGBoost, SHAP, TF-IDF, NLTK, Azure OpenAI-compatible chat endpoint |
| Public-safe review mode | Inspection-first. Full rerun requires restricted NH source tables that are not included. |

## Problem

ETF selection is hard to explain from return history alone. An ETF combines many holdings, and the signals that matter to an investor can include accumulated return, tracking error, Sharpe ratio, volatility, dividend behavior, sector or holding concentration, and the way customers already hold similar products.

Gen Pick frames the problem as a data-product question:

- cluster ETFs by interpretable quantitative indicators,
- pick representative ETFs that make each cluster easier to review,
- summarize the major holdings so a reviewer can understand what business exposure an ETF actually represents,
- connect holding-company descriptions to return-related keywords using SHAP.

The repository does not claim to predict future returns, produce investment advice, or guarantee portfolio performance.

## Role and Contribution

My primary contribution was the explainable ETF summary and evidence layer:

- designed the generative AI summary flow for ETF holdings,
- handled the prompt-size constraint by using the top holdings by weight instead of sending every holding description,
- designed an indirect validation approach for generated summaries using cluster consistency,
- built the TF-IDF, XGBoost, and SHAP keyword analysis that links company-description terms to return-related signals,
- organized public-safe outputs and documentation so the project can be reviewed without exposing restricted contest data.

The clustering and recommendation artifacts are presented as a team competition deliverable. This portfolio README separates my contribution from the broader team result.

## Approach

### 1. ETF indicator clustering

The analysis used 253 ETFs with 23 indicators. The indicators include return windows, accumulated return score, information ratio score, Sharpe score, current ratio score, tracking-error score, maximum drawdown score, volatility score, dividend frequency and amount, customer growth-account ratios, and top-holding concentration.

The workflow standardized the indicator scale, used t-SNE for cluster-structure inspection, and compared KMeans, Agglomerative Clustering, Spectral Clustering, and MeanShift. The public result table records KMeans with 4 clusters as the selected method:

| Method | Cluster count | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: |
| KMeans | 4 | 0.4343 | 266.6945 | 0.7371 |
| Agglomerative | 4 | 0.3976 | 236.8202 | 0.7463 |
| Spectral | 4 | 0.4336 | 261.3545 | 0.7418 |
| MeanShift | 2 | 0.3455 | 152.7156 | 1.1798 |

See [assets/presentation-slide-14-clustering.png](assets/presentation-slide-14-clustering.png) and [results/clustering_model_scores.csv](results/clustering_model_scores.csv).

### 2. Recommendation rationale

The recommendation surface is cluster-first rather than price-forecast-first. Each ETF is assigned to a cluster, and the representative ETF for a cluster is defined as the ETF closest to the cluster center. This makes the output easier to explain: a reviewer can inspect a small number of representative ETFs before comparing the full ETF universe.

An XGBoost classifier was trained to explain cluster membership. The top cluster-discriminating indicators in the public result file are accumulated return score, tracking-error score, Sharpe score, dividend frequency, and 1-year total return. See [results/cluster_feature_importance.csv](results/cluster_feature_importance.csv) and [assets/presentation-slide-15-feature-importance.png](assets/presentation-slide-15-feature-importance.png).

### 3. Generative AI ETF summary

The summary step turns ETF holdings into natural-language descriptions. Because large ETFs can contain many holdings, the prompt uses the top 30 holdings by portfolio weight. That choice preserves the dominant business exposure while keeping the input small enough for the chat endpoint.

The summary quality check was designed around consistency, not around unsupported human labels. Summaries from ETFs in the same cluster should expose similar investment themes and should not conflict with the cluster interpretation.

### 4. SHAP keyword evidence

The keyword workflow uses English company descriptions from ETF holdings, vectorizes them with TF-IDF, trains an XGBoost model against return-related targets, and uses SHAP values to identify words that contribute to the model output.

The public keyword evidence is in [results/keyword_importance_shap.csv](results/keyword_importance_shap.csv) and the slide image [assets/presentation-slide-22-shap-keywords.png](assets/presentation-slide-22-shap-keywords.png). The largest public keyword importances include terms such as `platform`, `segment`, `oper`, `develop`, `asset`, and `includ`. These are evidence features, not investment recommendations.

## Data and Public-Safety Boundary

The original analysis depended on NH-provided ETF and stock tables. Those source tables, personal or pledge documents, and contest-restricted data are not included here.

Public-safe files included in this repository:

- analysis and pipeline code,
- derived result CSVs for clustering, cluster explanation, and SHAP keyword evidence,
- selected presentation/report/demo assets already present in the repo,
- documentation describing the method and limits.

Excluded from publication:

- original NH source tables and raw contest data,
- NH security pledge PDFs, destruction pledge PDFs, or signed/private documents,
- raw bundles, archives, or Drive folders,
- `.env` files, credentials, private keys, and real service credentials,
- unsupported claims about investment returns or financial advice.

## Evidence and Results

Reviewer-visible evidence:

- [results/clustering_model_scores.csv](results/clustering_model_scores.csv): clustering model comparison.
- [results/cluster_metric_means.csv](results/cluster_metric_means.csv): cluster-level ETF indicator means.
- [results/cluster_feature_importance.csv](results/cluster_feature_importance.csv): XGBoost explanation of cluster assignment.
- [results/keyword_importance_shap.csv](results/keyword_importance_shap.csv): SHAP keyword importance from holding-description text.
- [results/sample_*_shap_values.csv](results/): sample per-holding SHAP outputs.
- [assets/presentation-slide-14-clustering.png](assets/presentation-slide-14-clustering.png), [assets/presentation-slide-15-feature-importance.png](assets/presentation-slide-15-feature-importance.png), and [assets/presentation-slide-22-shap-keywords.png](assets/presentation-slide-22-shap-keywords.png): visual evidence from the presentation.
- [assets/gen-pick-presentation.pdf](assets/gen-pick-presentation.pdf), [assets/analysis-report.pdf](assets/analysis-report.pdf), and [assets/gen-pick-demo.mp4](assets/gen-pick-demo.mp4): existing project artifacts.

The evidence supports the analysis workflow and public-safe outputs. It does not support a claim that the service improves realized investor returns.

## Reproducibility

Full end-to-end reproduction is blocked by data access: the raw NH source tables are not public and are intentionally excluded.

Public inspection path:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Then inspect:

- [notebooks/gen_pick_analysis.ipynb](notebooks/gen_pick_analysis.ipynb) for the original notebook flow,
- [src/clustering_pipeline.py](src/clustering_pipeline.py) for the clustering pipeline,
- [src/gen_pick_full_pipeline.py](src/gen_pick_full_pipeline.py) for the integrated clustering, summary, TF-IDF, XGBoost, and SHAP workflow,
- [results/](results/) for derived outputs that do not require restricted source tables.

The generative AI step requires a locally provided Azure OpenAI-compatible credential. Do not commit real credentials.

## Limitations

- Full rerun is not possible from this public repository because the NH source data is excluded.
- The clustering metrics evaluate structure in the available feature space; they do not prove investment suitability.
- The representative ETF selection is an explainability aid, not a personalized financial recommendation.
- Generated summaries were checked with cluster-consistency logic, not with a gold human-label dataset.
- SHAP keyword outputs explain the trained text model, not causal drivers of future ETF returns.
- Public assets should be reviewed again before any external publication because the original contest and Drive materials include restricted pledge and data documents.

## Repository Structure

```text
.
|-- README.md
|-- assets/
|   |-- analysis-report.pdf
|   |-- gen-pick-demo.mp4
|   |-- gen-pick-presentation.pdf
|   `-- presentation-slide-*.png
|-- docs/
|   |-- data-notice.md
|   |-- methodology.md
|   `-- portfolio-summary.md
|-- notebooks/
|   `-- gen_pick_analysis.ipynb
|-- results/
|   |-- clustering_model_scores.csv
|   |-- cluster_feature_importance.csv
|   |-- cluster_metric_means.csv
|   |-- keyword_importance_shap.csv
|   `-- sample_*_shap_values.csv
|-- src/
|   |-- clustering_pipeline.py
|   `-- gen_pick_full_pipeline.py
`-- requirements.txt
```
