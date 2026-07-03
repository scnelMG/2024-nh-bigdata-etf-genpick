# Methodology

This document describes the public-safe method behind Gen Pick. It is written for portfolio review, not as financial advice or a guarantee of investment outcomes.

## Data Inputs

The original competition workflow used NH-provided ETF and stock tables. The public repository keeps only derived artifacts and documentation.

Main input categories:

- ETF master and indicator data,
- dividend history and dividend frequency,
- customer holding and growth-account ratios,
- ETF holding composition,
- English company descriptions for holdings,
- derived return, risk, and concentration indicators.

The working clustering table contains 253 ETFs and 23 indicators. The raw tables are excluded from this repository because they are contest-restricted and may contain private or licensed data.

## ETF Indicator Clustering

The clustering layer turns a large ETF universe into reviewable investor-style groups.

Processing steps:

1. Clean missing and exceptional cases in the raw NH tables.
2. Convert dividend frequency into an annualized count.
3. Build the ETF feature matrix from return, risk, dividend, holding-concentration, and customer-ratio indicators.
4. Standardize indicators so scale differences do not dominate the model.
5. Use t-SNE for visual cluster-structure inspection.
6. Compare KMeans, Agglomerative Clustering, Spectral Clustering, and MeanShift.
7. Select the final method using Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.

Public model comparison:

| Method | Cluster count | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: |
| KMeans | 4 | 0.4343 | 266.6945 | 0.7371 |
| Agglomerative | 4 | 0.3976 | 236.8202 | 0.7463 |
| Spectral | 4 | 0.4336 | 261.3545 | 0.7418 |
| MeanShift | 2 | 0.3455 | 152.7156 | 1.1798 |

KMeans with 4 clusters is the selected public method because it has the strongest recorded Silhouette and Calinski-Harabasz scores and the lowest Davies-Bouldin score in the comparison table.

## Cluster Explanation

After assigning clusters, the workflow trains an XGBoost classifier with cluster label as the target. This is not used to predict investment performance. It is used to explain which indicators separate the clusters.

The strongest public feature importances are:

| Rank | Feature | Interpretation |
| ---: | --- | --- |
| 1 | `acl_pft_rt_z_sor` | accumulated return score |
| 2 | `trk_err_z_sor` | tracking-error score |
| 3 | `shpr_z_sor` | Sharpe score |
| 4 | `dividend_num` | dividend frequency |
| 5 | `yr1_tot_pft_rt` | 1-year total return |

The representative ETF for a cluster is defined as the ETF nearest to the cluster center. This makes the recommendation output easier to inspect and explain because each cluster has a concrete reference product.

## Generative AI ETF Summary

The generative AI layer summarizes the business exposure of an ETF from its holdings. The input is the holding weight and company description for the largest holdings.

The main implementation constraint was prompt size. Large ETFs can hold many companies, so the workflow limits the input to the top 30 holdings by portfolio weight. This is a practical tradeoff: it preserves the dominant exposure while keeping the request small enough for the chat endpoint.

The public repository does not include real credentials. The source code expects a locally supplied Azure OpenAI-compatible credential when the summary step is run.

## Summary Evaluation Strategy

The project did not have a gold-label summary dataset. Instead, it used an indirect consistency check:

- ETFs in the same cluster should show related themes in their generated summaries.
- Cluster descriptions should not conflict with the generated summaries.
- Summary terms should be explainable from high-weight holdings.

This is a pragmatic evaluation method for a competition prototype, but it is not a substitute for human financial-review validation.

## SHAP Keyword Evidence

The keyword layer connects ETF holdings text to return-related evidence.

Processing steps:

1. Clean English company descriptions.
2. Remove punctuation and stop words.
3. Vectorize descriptions with TF-IDF.
4. Train an XGBoost model for the return-related target.
5. Use SHAP values to identify terms that move the model output.

The public SHAP keyword file surfaces terms such as `platform`, `segment`, `oper`, `develop`, `asset`, and `includ`. These terms are evidence features from the trained model. They should be interpreted as model explanations, not as causal investment signals.

## Reproducibility Boundary

The public repository supports inspection and partial review:

- code structure in `src/`,
- original notebook in `notebooks/`,
- derived outputs in `results/`,
- presentation-derived visual evidence in `assets/`.

It does not support full rerun without the restricted NH source tables.

## Limitations

- Clustering quality metrics describe separation in the feature space, not investor utility.
- Representative ETF selection is an explanation strategy, not a personalized recommendation.
- The summary check is cluster-consistency based and does not prove factual completeness.
- SHAP values explain the model trained in this project; they do not prove causal drivers of future ETF returns.
- Data access, contest rules, and publish-safety requirements limit what can be included in the public repository.
