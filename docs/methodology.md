# Methodology

## 1. ETF Indicator Clustering

### Dataset

- ETF 수: 253개
- 지표 수: 23개
- 기간: 2024-05-28부터 2024-08-26까지 제공 데이터 기준
- 주요 테이블: ETF 점수 정보, ETF 배당 내역, 고객 보유 정보, ETF 구성 종목 정보

### Features

군집화에는 총수익률, 누적수익률 점수, 정보비율 점수, 샤프지수 점수, 상관관계 점수, 트래킹에러 점수, 최대낙폭 점수, 변동성 점수, 배당금액, 배당 주기, 고객 유형별 계좌 수 비율, 상위 5개 구성 종목 비중 합계를 사용했습니다.

### Modeling Process

1. 결측 및 이상 케이스를 정리했습니다.
2. Standard Scaling으로 지표 스케일을 맞췄습니다.
3. t-SNE로 차원을 축소해 군집 구조를 확인했습니다.
4. K-means, 병합 군집화, 스펙트럼 군집화, MeanShift를 비교했습니다.
5. Elbow method로 후보 군집 수를 정했습니다.
6. Silhouette, Calinski-Harabasz, Davies-Bouldin 지표로 최종 알고리즘을 선택했습니다.

### Result

| Method | Cluster Count | Silhouette | Calinski-Harabasz | Davies-Bouldin |
| --- | ---: | ---: | ---: | ---: |
| KMeans | 4 | 0.4343 | 266.6945 | 0.7371 |
| Agglomerative | 4 | 0.3976 | 236.8202 | 0.7463 |
| Spectral | 4 | 0.4336 | 261.3545 | 0.7418 |
| MeanShift | 2 | 0.3455 | 152.7156 | 1.1798 |

K-means가 Silhouette과 Calinski-Harabasz에서 가장 높고 Davies-Bouldin에서도 가장 낮아 최종 모델로 선택됐습니다.

## 2. Cluster Explainability

군집 결과를 종속변수로 두고 XGBoost 분류 모델을 학습했습니다. 이후 Feature Importance로 어떤 지표가 군집 구분에 크게 기여했는지 확인했습니다.

상위 지표는 다음과 같습니다.

| Rank | Feature | Interpretation |
| ---: | --- | --- |
| 1 | acl_pft_rt_z_sor | 누적수익률 점수 |
| 2 | trk_err_z_sor | 트래킹에러 점수 |
| 3 | shpr_z_sor | 샤프지수 점수 |
| 4 | dividend_total | 배당금액 |
| 5 | yr1_tot_pft_rt | 1년 총수익률 |

## 3. Generative AI ETF Summary

ETF 구성 종목의 영문 사업 개요와 구성 비율을 입력으로 사용해 ETF 전체 특성을 요약했습니다. 모델은 GPT-4o-mini를 사용했습니다.

### Token Limit Strategy

모든 구성 종목 설명을 입력하면 토큰 제한이 발생할 수 있기 때문에 구성 비율 기준 상위 30개 종목만 입력했습니다. ETF의 성격은 보통 비중이 높은 구성 종목이 결정하므로, 정보 손실과 입력 가능성 사이의 균형을 맞춘 방식입니다.

### Evaluation Strategy

생성형 AI 요약에는 명확한 정답 레이블이 없기 때문에 직접적인 정량 평가는 어려웠습니다. 대신 군집 결과와 요약 결과의 정합성을 간접 검증 기준으로 삼았습니다.

- 같은 군집 ETF의 요약이 유사한 산업 키워드와 투자 방향성을 보이는지 확인했습니다.
- 군집별 해석과 생성 요약이 충돌하지 않는지 점검했습니다.
- 요약 결과가 ETF 구성 종목의 상위 비중 기업 특성을 반영하는지 확인했습니다.

## 4. SHAP Keyword Curation

ETF 구성 종목 사업 개요 텍스트와 평균 수익률을 연결해 수익성과 관련된 키워드를 도출했습니다.

1. 영문 사업 개요에서 영어 외 문자 제거, 불용어 제거, 품사 필터링, 어간 추출을 수행했습니다.
2. TF-IDF로 텍스트를 벡터화했습니다.
3. XGBoost 회귀 모델로 평균 수익률을 예측했습니다.
4. SHAP 값으로 수익성에 영향을 미치는 키워드를 해석했습니다.

이 접근은 ETF가 어떤 종목으로 구성돼 있는지를 설명하는 것을 넘어, 어떤 사업 키워드가 수익성과 연결되는지 보여주기 위한 분석입니다.
