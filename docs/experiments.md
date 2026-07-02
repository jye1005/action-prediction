# Experiment Log

AI 코딩 에이전트 다음 행동 예측 실험 결과 기록이다.

## Score Terms

- `Local CV`: 로컬 train 데이터를 나눠 검증한 Macro-F1.
- `Public LB`: 대회 서버 public leaderboard Macro-F1.
- `Artifact`: 로컬 제출 zip 또는 공유 링크.

## Results

| Date | Model / Method | Input | Validation | Local CV | Public LB | Artifact | Note |
| --- | --- | --- | --- | ---: | ---: | --- | --- |
| 2026-07-01 | TF-IDF + Logistic Regression | `current_prompt` only | stratified holdout 20%, seed 42 | 0.4383 | - | `model/tfidf_logreg.pkl` | 배포 baseline을 `.py`로 변환해 실행 |
| 2026-07-01 | `intfloat/multilingual-e5-small` fine-tuning | `current_prompt` + recent history + session/workspace meta | stratified holdout 15%, seed 42 | 0.49345 | - | `model/e5-small-router` | 초기 E5 실험 |
| 2026-07-01 | `intfloat/multilingual-e5-small` fine-tuning | `current_prompt` + recent history + session/workspace meta | stratified holdout 20%, seed 42 | 0.48948 | - | `submissions/submit_e5-small-val20_f1-0.48948_20260701.zip` | baseline과 같은 split 비율로 비교 |
| 2026-07-01 | `ibm-granite/granite-embedding-311m-multilingual-r2` fine-tuning | `[META] + [HIST] + [CUR]`, recent 6 user-action pairs | GroupKFold 5, fold0, session id group | 0.73255 | - | `submissions/submit_granite-311m-fold0_f1-0.73255_20260701.zip` | ModernBERT 기반 Granite 재현 |
| 2026-07-01 | `ibm-granite/granite-embedding-311m-multilingual-r2` fine-tuning + logit bias tuning | same as above | GroupKFold 5, fold0, session id group | 0.73697 | 0.73078 | `submissions/submit_granite-311m-fold0_bias_f1-0.73697_20260701.zip` / [Drive](https://drive.google.com/file/d/1nIw48xZB1kVZmsO1v3o18E_3TR2NCQmu/view?usp=drive_link) | Current SOTA |

## Current Best

Current best submission candidate:

```text
submissions/submit_granite-311m-fold0_bias_f1-0.73697_20260701.zip
```

Summary:

- Base model: `ibm-granite/granite-embedding-311m-multilingual-r2`
- Local CV: `0.73697`
- Public LB: `0.73078`
- Method: fine-tuning + class logit bias tuning
- Model size: about `629M` unpacked
- Submit zip size: about `496M`

## Notes

- E5-small improved over the TF-IDF baseline, but Granite gave a much larger jump.
- Granite requires `transformers==4.48.3` because it loads as `ModernBertForSequenceClassification`.
- The evaluation server default is `transformers==4.46.3`; Granite submissions should include a pinned `requirements.txt`.
- Logit bias tuning improved fold0 Macro-F1 from `0.732565` to `0.736965` without retraining.

