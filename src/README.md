# Source Code

이 폴더는 대회 당시 분석 흐름을 포트폴리오용으로 정리한 코드입니다.

## Files

| File | Description |
| --- | --- |
| `clustering_pipeline.py` | ETF 지표 병합, 전처리, 군집화 실험 중심 코드 |
| `gen_pick_full_pipeline.py` | 군집화, 생성형 AI 요약, TF-IDF/XGBoost/SHAP 키워드 도출 통합 코드 |

## Running Notes

원본 NH투자증권 제공 테이블은 공개 저장소에 포함하지 않았기 때문에, `src/` 코드는 raw data가 있는 로컬 환경에서만 전체 재실행할 수 있습니다. 공개 저장소에서는 `results/`의 파생 결과물로 분석 결과를 확인할 수 있습니다.

생성형 AI 요약 구간을 실행하려면 Azure OpenAI 키를 환경변수로 설정해야 합니다.

```bash
AZURE_OPENAI_API_KEY=your_key
```

엔드포인트를 바꾸려면 `AZURE_OPENAI_ENDPOINT`를 함께 설정합니다.
