# AI Agent Action Prediction

AI 코딩 에이전트 세션 상태에서 다음 행동(action)을 14개 클래스 중 하나로 예측하는 대회용 작업 저장소다.

## Baseline

현재 코드는 배포 baseline notebook을 Python script로 변환한 초기 버전이다.

- `scripts/baseline_train.py`: `current_prompt`만 사용해 TF-IDF + Logistic Regression 모델을 학습하고 `model/tfidf_logreg.pkl`로 저장한다.
- `scripts/baseline_inference.py`: 저장된 모델을 불러와 `data/test.jsonl`을 예측하고 `output/submission.csv`를 생성한다.

## Expected Data Layout

실행 시 데이터는 저장소 루트 기준 아래 위치에 둔다. 데이터 파일은 용량과 대회 규정상 git에 포함하지 않는다.

```text
data/
  train.jsonl
  train_labels.csv
  test.jsonl
  sample_submission.csv
```

## Action Classes

`read_file`, `grep_search`, `list_directory`, `glob_pattern`, `edit_file`, `write_file`, `apply_patch`, `run_bash`, `run_tests`, `lint_or_typecheck`, `ask_user`, `plan_task`, `web_search`, `respond_only`

## Run

```bash
python scripts/baseline_train.py
python scripts/baseline_inference.py
```

학습 결과와 추론 결과는 각각 아래에 생성된다.

```text
model/tfidf_logreg.pkl
output/submission.csv
```

