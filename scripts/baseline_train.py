import csv
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

# 예측 대상 14개 클래스 (Macro-F1 계산에 사용)
ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

DATA_DIR = "./data"

# train.jsonl: 한 줄 = 샘플 하나
samples = [json.loads(line)
           for line in open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8")
           if line.strip()]

# train_labels.csv: id -> action 매핑
labels = {row["id"]: row["action"]
          for row in csv.DictReader(open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8"))}

# 입력 X = current_prompt, 정답 y = action
X = [s["current_prompt"] for s in samples]
y = [labels[s["id"]] for s in samples]

print("samples:", len(X), "| classes:", len(set(y)))

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42,
)
print("train:", len(X_train), "| val:", len(X_val))

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=80_000,
        sublinear_tf=True, lowercase=True,
    )),
    ("clf", LogisticRegression(
        max_iter=500, class_weight="balanced", C=2.0,
    )),
])

pipe.fit(X_train, y_train)
print("학습 완료")

val_pred = pipe.predict(X_val)
macro_f1 = f1_score(y_val, val_pred, labels=ALL_CLASSES, average="macro", zero_division=0)
print(f"Validation Macro-F1: {macro_f1:.4f}")

# 전체 학습 데이터로 재학습
pipe.fit(X, y)

# 저장 (추론용 script.py가 ./model/tfidf_logreg.pkl 을 불러옵니다)
os.makedirs("./model", exist_ok=True)
joblib.dump(pipe, "./model/tfidf_logreg.pkl", compress=3)
print("저장 완료: ./model/tfidf_logreg.pkl")
