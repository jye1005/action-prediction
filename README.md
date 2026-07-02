# AI Agent Action Prediction

AI 코딩 에이전트 세션 상태에서 다음 행동(action)을 14개 클래스 중 하나로 예측하는 대회용 작업 저장소다.

## Experiments

시도한 모델과 성능 기록은 `docs/experiments.md`에 정리한다.

## Baseline

현재 코드는 배포 baseline notebook을 Python script로 변환한 초기 버전이다.

- `scripts/baseline_train.py`: `current_prompt`만 사용해 TF-IDF + Logistic Regression 모델을 학습하고 `model/tfidf_logreg.pkl`로 저장한다.
- `scripts/baseline_inference.py`: 저장된 모델을 불러와 `data/test.jsonl`을 예측하고 `output/submission.csv`를 생성한다.

## Transformer Experiment

`intfloat/multilingual-e5-small` 기반 action router 실험을 추가했다. IBM Granite embedding 모델은 제외하고, 먼저 다국어 E5-small을 sequence classification으로 fine-tuning하는 구성이다.

선택 이유:

- 한국어/영어/mixed 입력을 모두 처리할 수 있다.
- small급 모델이라 T4 16GB, hidden test 30,000건 추론 제한 안에서 운영하기 쉽다.
- `current_prompt`만 쓰는 baseline보다 `history`, `session_meta`, workspace 상태를 함께 넣을 수 있다.

구성:

- `src/action_router/features.py`: JSONL 샘플을 모델 입력 텍스트로 렌더링한다.
- `scripts/train_e5_router.py`: E5-small sequence classifier fine-tuning.
- `scripts/infer_e5_router.py`: fine-tuned 모델로 `submission.csv` 생성.
- `requirements-transformer.txt`: transformer 실험용 의존성.

입력 텍스트에는 다음 정보를 포함한다.

- `current_prompt`
- 최근 history의 user 발화와 assistant action/result
- `user_tier`, `language_pref`, token budget bucket, turn index, elapsed bucket
- workspace language mix, LOC, git dirty, open files, last CI status

학습:

```bash
pip install -r requirements-transformer.txt
python scripts/train_e5_router.py \
  --data-dir ./data \
  --model-name intfloat/multilingual-e5-small \
  --output-dir ./model/e5-small-router \
  --epochs 3 \
  --batch-size 16 \
  --max-length 512 \
  --max-history 8
```

추론:

```bash
python scripts/infer_e5_router.py \
  --data-dir ./data \
  --model-dir ./model/e5-small-router \
  --output-path ./output/submission.csv \
  --batch-size 64
```

학습된 모델 디렉터리는 제출 시 `model/e5-small-router` 형태로 포함하고, 평가 서버에서는 인터넷 없이 로컬 모델만 로드해야 한다.

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
