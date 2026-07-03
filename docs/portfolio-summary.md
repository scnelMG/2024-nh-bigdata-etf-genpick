# Portfolio Summary

## Project

Gen Pick is an ETF recommendation and explanation prototype built for the 2024 NH Investment & Securities Big Data Competition. It uses ETF indicators to cluster products, selects representative ETFs for review, summarizes ETF holdings with generative AI, and uses SHAP keyword evidence to explain holding-description signals.

## Problem

ETF review is difficult because a single product can represent many holdings and multiple investment signals at once. A reviewer may need to understand return history, risk, dividend behavior, customer holding patterns, and business exposure before deciding whether an ETF fits a use case.

Gen Pick reframes the task as an explainable data-product workflow rather than a black-box return forecast.

## Role and Contribution

My main contribution was the generative AI and evidence layer:

- designed the ETF holdings summary flow,
- handled prompt-size limits by selecting the top 30 holdings by weight,
- created a cluster-consistency check for generated summaries,
- connected company-description text to return-related evidence with TF-IDF, XGBoost, and SHAP,
- prepared the public-safe documentation and review path.

The clustering and final product concept were team competition outputs.

## Technical Decisions

Important decisions:

- use 23 ETF indicators across return, risk, dividend, holding, and customer-ratio dimensions,
- compare four clustering algorithms and select KMeans with 4 clusters based on public metric outputs,
- use cluster-center distance to define representative ETFs,
- use XGBoost feature importance to explain cluster membership,
- use top-weight holdings for generative summaries to control request size,
- use SHAP keyword importance to expose text evidence from holding descriptions.

## Evidence

Public evidence lives in:

- `results/clustering_model_scores.csv`,
- `results/cluster_metric_means.csv`,
- `results/cluster_feature_importance.csv`,
- `results/keyword_importance_shap.csv`,
- `results/sample_*_shap_values.csv`,
- `assets/presentation-slide-14-clustering.png`,
- `assets/presentation-slide-15-feature-importance.png`,
- `assets/presentation-slide-22-shap-keywords.png`.

The evidence supports the analysis and explanation workflow. It does not claim future investment performance.

## Reproducibility

This is an inspection-first public repository. Full rerun requires restricted NH source tables that are intentionally excluded.

Reviewers can inspect:

- the notebook in `notebooks/gen_pick_analysis.ipynb`,
- the pipeline code in `src/`,
- the derived result files in `results/`,
- the method notes in `docs/methodology.md`.

## Public-Safety Notes

Do not publish NH security or destruction pledge PDFs, raw contest data, raw bundles, `.env` files, credentials, or copied Drive folders. The public repository should remain limited to reviewed code, derived outputs, and safe presentation artifacts.

## Limitations

- No public raw data is included.
- Generated summaries were evaluated by consistency checks, not by a gold summary benchmark.
- SHAP keyword evidence explains a model trained on available text features; it is not causal financial evidence.
- The project is not investment advice and should not be used as a live trading or advisory system.
