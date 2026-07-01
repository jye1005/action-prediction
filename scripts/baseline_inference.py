import csv
import json
import os

import joblib

REQUIRED_KEYS = ("id", "session_meta", "history", "current_prompt")


def load_jsonl(path):
    """평가 데이터(jsonl) 로드. 한 줄당 샘플 하나."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no} JSON 파싱 실패: {e}")
            samples.append(obj)
    return samples


def validate_samples(samples):
    """필수 키 존재 여부 검증 (학습 데이터와 동일 스키마)."""
    n_bad = 0
    for s in samples:
        for k in REQUIRED_KEYS:
            if k not in s:
                n_bad += 1
                break
    if n_bad:
        print(f" 경고: 필수 키 누락 샘플 {n_bad}건 (빈 텍스트로 처리)")
    return n_bad


def extract_text(sample):
    """모델 입력 텍스트 추출 — 학습 때와 동일하게 current_prompt만 사용."""
    text = sample.get("current_prompt", "")
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    return text


def build_features(samples):
    """샘플 리스트 → (ids, 모델 입력 텍스트 리스트)."""
    ids = []
    texts = []
    for s in samples:
        ids.append(s.get("id", ""))
        texts.append(extract_text(s))
    n_empty = sum(1 for t in texts if not t.strip())
    if n_empty:
        print(f" 경고: current_prompt가 비어있는 샘플 {n_empty}건")
    return ids, texts

def load_sample_submission(path):
    """sample_submission.csv 로드 — 제출 파일의 id 순서/컬럼 기준."""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if fieldnames is None or fieldnames[:2] != ["id", "action"]:
        raise ValueError(
            f"sample_submission 컬럼이 (id, action)이 아님: {fieldnames}")
    return fieldnames, rows


def merge_predictions(sub_rows, ids, preds):
    """sample_submission의 id 순서에 맞춰 예측값 병합."""
    pred_map = dict(zip(ids, preds))
    n_missing = 0
    for row in sub_rows:
        p = pred_map.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder를 유지한 id {n_missing}건")
    return sub_rows


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def main():
    # ---- 경로 변수 (필요에 따라 수정) ----
    TEST_DIR = "./data"            # test.jsonl, sample_submission.csv 위치
    MODEL_DIR = "./model"          # tfidf_logreg.pkl 위치
    OUT_DIR = "./output"
    TEST_PATH = os.path.join(TEST_DIR, "test.jsonl")
    SAMPLE_SUB_PATH = os.path.join(TEST_DIR, "sample_submission.csv")
    MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg.pkl")
    OUT_PATH = os.path.join(OUT_DIR, "submission.csv")

    # ---- 모델 로드 ----
    print("Load model...")
    model = joblib.load(MODEL_PATH)
    classes = list(getattr(model, "classes_", []))
    print(f" OK. classes={len(classes)}")

    # ---- 테스트 데이터 로드 ----
    print("Load test data...")
    samples = load_jsonl(TEST_PATH)
    validate_samples(samples)
    print(f" samples={len(samples)}")

    # ---- 전처리 (학습과 동일: current_prompt 추출) ----
    print("Build features...")
    ids, texts = build_features(samples)
    print(f" texts={len(texts)}")

    # ---- 예측 ----
    print("Inference model...")
    preds = model.predict(texts) if texts else []
    preds = [str(p) for p in preds]
    print(f" preds={len(preds)}")

    # ---- sample_submission 기반 결과 생성 (action 컬럼에 예측 클래스 채움) ----
    print("Build submission...")
    fieldnames, sub_rows = load_sample_submission(SAMPLE_SUB_PATH)
    sub_rows = merge_predictions(sub_rows, ids, preds)
    save_submission(OUT_PATH, fieldnames, sub_rows)
    print(f"Saved: {OUT_PATH} (rows={len(sub_rows)})")

if __name__ == "__main__":
    main()
