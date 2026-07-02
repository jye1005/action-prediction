from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES
from action_router.features import render_sample


def load_ml_dependencies():
    global DictVectorizer
    global LogisticRegression
    global StratifiedKFold
    global classification_report
    global f1_score
    global csr_matrix
    global hstack
    global joblib
    global np
    global train_test_split

    import joblib
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import StratifiedKFold, train_test_split


PROMPT_KEYWORDS = {
    "read": ["read", "open", "show", "view", "봐", "열어", "확인"],
    "search": ["grep", "search", "find", "찾", "검색"],
    "edit": ["edit", "fix", "change", "update", "patch", "수정", "고쳐", "바꿔"],
    "write": ["write", "create", "add", "만들", "추가", "작성"],
    "run": ["run", "execute", "fire up", "돌려", "실행"],
    "test": ["test", "spec", "pytest", "happy path", "테스트"],
    "lint": ["lint", "typecheck", "type check", "mypy", "eslint", "타입"],
    "plan": ["plan", "steps", "break down", "쪼개", "계획"],
    "ask": ["ask", "question", "clarify", "물어"],
    "web": ["web", "google", "internet", "browser", "검색해"],
}

RESULT_KEYWORDS = [
    "pass",
    "fail",
    "error",
    "patched",
    "read",
    "matches",
    "listed",
    "saved",
    "installed",
    "timeout",
    "warning",
    "lint",
    "test",
]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def resolve_device(requested):
    if requested != "auto":
        return requested
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def set_cache_env(cache_dir):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", str(Path(cache_dir) / "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(Path(cache_dir) / "sentence_transformers"))


def safe_name(value):
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_").lower()


def embedding_cache_signature(args, n_samples):
    return {
        "model_name": args.model_name,
        "max_history": args.max_history,
        "prefix": args.prefix,
        "max_length": args.max_length,
        "n_samples": n_samples,
    }


def auto_embedding_cache_path(args, signature):
    if args.embedding_cache:
        return args.embedding_cache
    digest = hashlib.md5(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    model_name = safe_name(args.model_name.split("/")[-1])
    return str(Path(args.cache_root) / f"{model_name}_router_{digest}.npy")


def load_embedding_cache_if_valid(cache_path, expected_signature):
    meta_path = f"{cache_path}.meta.json"
    if not Path(cache_path).exists() or not Path(meta_path).exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        actual_signature = json.load(f)
    if actual_signature != expected_signature:
        print("embedding cache signature mismatch, recompute instead.", flush=True)
        return None
    return np.load(cache_path)


def save_embedding_cache(cache_path, embeddings, signature):
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, embeddings)
    with open(f"{cache_path}.meta.json", "w", encoding="utf-8") as f:
        json.dump(signature, f, ensure_ascii=False, indent=2, sort_keys=True)


def encode_texts(args, texts):
    from sentence_transformers import SentenceTransformer

    device = resolve_device(args.device)
    print(f"device={device}", flush=True)
    model = SentenceTransformer(
        args.model_name,
        device=device,
        cache_folder=args.cache_dir or None,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = args.max_length
    embeddings = model.encode(
        [args.prefix + text for text in texts],
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=not args.no_progress,
        convert_to_numpy=True,
    )
    if args.save_encoder:
        encoder_dir = Path(args.output_dir) / "embedding_model"
        if encoder_dir.exists():
            shutil.rmtree(encoder_dir)
        model.save(str(encoder_dir))
        print(f"saved encoder: {encoder_dir}", flush=True)
    return embeddings.astype(np.float32, copy=False)


def bucket_number(value, cuts, prefix):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{prefix}=missing"
    for cut in cuts:
        if number <= cut:
            return f"{prefix}<={int(cut)}"
    return f"{prefix}>{int(cuts[-1])}"


def recent_actions(history, max_items):
    actions = []
    for item in list(history)[-max_items:]:
        if item.get("role") == "assistant_action":
            actions.append(str(item.get("name", "")))
    return [action for action in actions if action]


def last_action_result(history):
    for item in reversed(list(history)):
        if item.get("role") == "assistant_action":
            return str(item.get("result_summary", ""))
    return ""


def build_manual_feature_dict(sample, max_history):
    features = {}
    prompt = sample.get("current_prompt", "")
    if not isinstance(prompt, str):
        prompt = "" if prompt is None else str(prompt)
    history = sample.get("history", [])
    if not isinstance(history, list):
        history = []

    features[bucket_number(len(history), [0, 2, 5, 10, 20], "history_len")] = 1.0
    features[bucket_number(len(prompt), [0, 40, 100, 200, 400], "prompt_len")] = 1.0
    lower_prompt = prompt.lower()
    for group, keywords in PROMPT_KEYWORDS.items():
        if any(keyword in lower_prompt for keyword in keywords):
            features[f"prompt_kw={group}"] = 1.0

    actions = recent_actions(history, max_history)
    if actions:
        features[f"last_action={actions[-1]}"] = 1.0
        for pos, action in enumerate(reversed(actions[-3:]), start=1):
            features[f"recent_action_{pos}={action}"] = 1.0
        for left, right in zip(actions, actions[1:]):
            features[f"action_bigram={left}>{right}"] = features.get(f"action_bigram={left}>{right}", 0.0) + 1.0

    result = last_action_result(history).lower()
    for keyword in RESULT_KEYWORDS:
        if keyword in result:
            features[f"result_kw={keyword}"] = 1.0

    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") if isinstance(meta, dict) else {}
    workspace = workspace if isinstance(workspace, dict) else {}
    features[f"tier={meta.get('user_tier', 'missing')}"] = 1.0
    features[f"lang_pref={meta.get('language_pref', 'missing')}"] = 1.0
    features[f"last_ci={workspace.get('last_ci_status', 'missing')}"] = 1.0
    features[f"git_dirty={workspace.get('git_dirty', 'missing')}"] = 1.0
    features[bucket_number(meta.get("turn_index"), [1, 3, 6, 10, 20], "turn")] = 1.0
    features[bucket_number(meta.get("budget_tokens_remaining"), [10_000, 30_000, 80_000, 150_000], "budget")] = 1.0
    features[bucket_number(meta.get("elapsed_session_sec"), [60, 300, 900, 1800, 3600], "elapsed")] = 1.0

    open_files = workspace.get("open_files", [])
    if isinstance(open_files, list):
        features[bucket_number(len(open_files), [0, 1, 3, 6, 10], "open_files")] = 1.0
        for file_path in open_files[-5:]:
            ext = os.path.splitext(str(file_path))[1].lower() or "no_ext"
            features[f"open_ext={ext}"] = features.get(f"open_ext={ext}", 0.0) + 1.0
    return features


def build_design_matrix(embeddings, samples, vectorizer=None, fit=False, use_manual_features=True, max_history=8, manual_weight=1.0):
    if not use_manual_features:
        return embeddings, None
    feature_dicts = [build_manual_feature_dict(sample, max_history) for sample in samples]
    if fit:
        vectorizer = DictVectorizer(sparse=True)
        manual = vectorizer.fit_transform(feature_dicts)
    else:
        manual = vectorizer.transform(feature_dicts)
    manual = manual.astype(np.float32) * float(manual_weight)
    return hstack([csr_matrix(embeddings), manual], format="csr"), vectorizer


def fit_logreg(x_train, y_train, seed, c):
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(x_train, y_train)
    return clf


def align_log_proba(clf, x, classes):
    proba = clf.predict_proba(x)
    aligned = np.full((x.shape[0], len(classes)), 1e-12, dtype=np.float64)
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for src_idx, label in enumerate(clf.classes_):
        aligned[:, class_to_idx[label]] = proba[:, src_idx]
    return np.log(np.clip(aligned, 1e-12, 1.0))


def predict_from_logits(logits, classes, bias=None):
    if bias is not None:
        logits = logits + bias.reshape(1, -1)
    return np.asarray(classes, dtype=object)[np.argmax(logits, axis=1)]


def tune_bias(logits, y_true, classes, rounds, span, steps):
    bias = np.zeros(len(classes), dtype=np.float64)
    grid = np.linspace(-span, span, steps)
    best = f1_score(y_true, predict_from_logits(logits, classes, bias), labels=list(classes), average="macro", zero_division=0)
    for _ in range(rounds):
        improved = False
        for class_idx in range(len(classes)):
            current = bias[class_idx]
            best_value = current
            for delta in grid:
                bias[class_idx] = current + delta
                score = f1_score(y_true, predict_from_logits(logits, classes, bias), labels=list(classes), average="macro", zero_division=0)
                if score > best:
                    best = score
                    best_value = bias[class_idx]
                    improved = True
            bias[class_idx] = best_value
        if not improved:
            break
    return bias, best


def fit_action_model(embeddings, samples, y_train, args, seed):
    x_train, vectorizer = build_design_matrix(
        embeddings,
        samples,
        fit=True,
        use_manual_features=not args.no_manual_features,
        max_history=args.max_history,
        manual_weight=args.manual_weight,
    )
    return {"clf": fit_logreg(x_train, y_train, seed, args.c), "feature_vectorizer": vectorizer}


def action_model_log_proba(model_bundle, embeddings, samples, classes, args):
    x, _ = build_design_matrix(
        embeddings,
        samples,
        vectorizer=model_bundle.get("feature_vectorizer"),
        fit=False,
        use_manual_features=not args.no_manual_features,
        max_history=args.max_history,
        manual_weight=args.manual_weight,
    )
    return align_log_proba(model_bundle["clf"], x, classes)


def run_holdout(embeddings, samples, y, args):
    train_idx, val_idx = train_test_split(np.arange(len(y)), test_size=args.val_size, stratify=y, random_state=args.seed)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    model = fit_action_model(embeddings[train_idx], train_samples, y[train_idx], args, args.seed)
    val_logits = action_model_log_proba(model, embeddings[val_idx], val_samples, ACTION_CLASSES, args)
    base_pred = predict_from_logits(val_logits, ACTION_CLASSES)
    base_f1 = f1_score(y[val_idx], base_pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    bias, tuned_f1 = tune_bias(val_logits, y[val_idx], ACTION_CLASSES, args.bias_rounds, args.bias_span, args.bias_steps)
    tuned_pred = predict_from_logits(val_logits, ACTION_CLASSES, bias)
    print(classification_report(y[val_idx], tuned_pred, labels=ACTION_CLASSES, zero_division=0, digits=4))
    return [model], bias, base_f1, tuned_f1


def run_cv(embeddings, samples, y, args):
    splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_logits = np.zeros((len(y), len(ACTION_CLASSES)), dtype=np.float64)
    models = []
    for fold, (train_idx, val_idx) in enumerate(splitter.split(embeddings, y), start=1):
        print(f"\n[fold {fold}/{args.folds}] train={len(train_idx)} val={len(val_idx)}", flush=True)
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        model = fit_action_model(embeddings[train_idx], train_samples, y[train_idx], args, args.seed + fold)
        models.append(model)
        oof_logits[val_idx] = action_model_log_proba(model, embeddings[val_idx], val_samples, ACTION_CLASSES, args)
        fold_pred = predict_from_logits(oof_logits[val_idx], ACTION_CLASSES)
        fold_f1 = f1_score(y[val_idx], fold_pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
        print(f"fold Macro-F1: {fold_f1:.6f}", flush=True)
    base_pred = predict_from_logits(oof_logits, ACTION_CLASSES)
    base_f1 = f1_score(y, base_pred, labels=ACTION_CLASSES, average="macro", zero_division=0)
    bias, tuned_f1 = tune_bias(oof_logits, y, ACTION_CLASSES, args.bias_rounds, args.bias_span, args.bias_steps)
    tuned_pred = predict_from_logits(oof_logits, ACTION_CLASSES, bias)
    print(classification_report(y, tuned_pred, labels=ACTION_CLASSES, zero_division=0, digits=4))
    return models, bias, base_f1, tuned_f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--output-dir", default="./model/embedding-logreg-router")
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--prefix", default="")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--embedding-cache", default="")
    parser.add_argument("--cache-root", default="./cache")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=2.0)
    parser.add_argument("--no-manual-features", action="store_true")
    parser.add_argument("--manual-weight", type=float, default=1.0)
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--bias-rounds", type=int, default=3)
    parser.add_argument("--bias-span", type=float, default=1.0)
    parser.add_argument("--bias-steps", type=int, default=41)
    parser.add_argument("--save-encoder", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    load_ml_dependencies()
    set_cache_env(args.cache_dir)

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    labels = load_labels(Path(args.data_dir) / "train_labels.csv")
    texts = [render_sample(sample, max_history=args.max_history) for sample in samples]
    y = np.asarray([labels[sample["id"]] for sample in samples], dtype=object)
    print(f"samples={len(texts)} classes={len(set(y))} model={args.model_name}", flush=True)

    signature = embedding_cache_signature(args, len(texts))
    embedding_cache = auto_embedding_cache_path(args, signature)
    print(f"embedding cache path: {embedding_cache}", flush=True)
    embeddings = load_embedding_cache_if_valid(embedding_cache, signature)
    if embeddings is None:
        embeddings = encode_texts(args, texts)
        save_embedding_cache(embedding_cache, embeddings, signature)
        print(f"saved embedding cache: {embedding_cache}", flush=True)
    print(f"embeddings={embeddings.shape}", flush=True)

    if args.cv:
        fold_models, bias, base_f1, tuned_f1 = run_cv(embeddings, samples, y, args)
    else:
        fold_models, bias, base_f1, tuned_f1 = run_holdout(embeddings, samples, y, args)

    print("\n=== Local validation ===")
    print(f"base Macro-F1:       {base_f1:.6f}")
    print(f"bias tuned Macro-F1: {tuned_f1:.6f}")
    print(f"gain:                {tuned_f1 - base_f1:+.6f}")
    print("bias:", dict(zip(ACTION_CLASSES, np.round(bias, 4).tolist())))

    print("\nFit full-data classifier...", flush=True)
    full_model = fit_action_model(embeddings, samples, y, args, args.seed)
    artifact = {
        "classes": ACTION_CLASSES,
        "model_name": args.model_name,
        "max_history": args.max_history,
        "prefix": args.prefix,
        "max_length": args.max_length,
        "normalize_embeddings": True,
        "use_manual_features": not args.no_manual_features,
        "manual_weight": args.manual_weight,
        "bias": bias,
        "fold_models": fold_models,
        "full_model": full_model,
        "use_full_model": True,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_dir / "embedding_logreg_artifact.joblib", compress=3)
    print(f"saved: {output_dir / 'embedding_logreg_artifact.joblib'}", flush=True)


if __name__ == "__main__":
    main()
