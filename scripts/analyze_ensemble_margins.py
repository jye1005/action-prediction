import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, LABEL2ID
from action_router.features import render_granite_sample, render_sample
from infer_ensemble_router import predict_logits_for_model


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def session_group(sample_id):
    return str(sample_id).split("-step_", 1)[0]


def render_texts(samples, feature_mode, max_history, max_history_events):
    if feature_mode == "sample":
        return [render_sample(sample, max_history=max_history) for sample in samples]
    if feature_mode == "granite":
        return [render_granite_sample(sample, max_history_events=max_history_events) for sample in samples]
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def print_bucket_stats(name, values, correct, thresholds, lower_is_uncertain=True):
    print(f"\n[{name} buckets]")
    for threshold in thresholds:
        mask = values <= threshold if lower_is_uncertain else values >= threshold
        count = int(mask.sum())
        if count == 0:
            print(f"{name} {'<=' if lower_is_uncertain else '>='} {threshold:.3f}: rows=0")
            continue
        acc = float(correct[mask].mean())
        print(f"{name} {'<=' if lower_is_uncertain else '>='} {threshold:.3f}: rows={count} acc={acc:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dirs", required=True)
    parser.add_argument("--feature-mode", choices=["granite", "sample"], default="granite")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--fold",
        type=int,
        default=0,
        help="When --model-dirs has one entry, score only this fold's validation split.",
    )
    parser.add_argument("--use-logit-bias", action="store_true", default=True)
    parser.add_argument("--no-logit-bias", dest="use_logit_bias", action="store_false")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    model_dirs = split_csv(args.model_dirs)
    single_fold_mode = len(model_dirs) == 1
    if not single_fold_mode and len(model_dirs) != args.folds:
        raise ValueError("--model-dirs must contain one model per fold, or exactly one model with --fold.")

    samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
    labels = load_labels(Path(args.data_dir) / "train_labels.csv")
    ids = [sample["id"] for sample in samples]
    y = np.asarray([LABEL2ID[labels[sample_id]] for sample_id in ids], dtype=np.int64)
    groups = np.asarray([session_group(sample_id) for sample_id in ids], dtype=object)
    texts = np.asarray(render_texts(samples, args.feature_mode, args.max_history, args.max_history_events), dtype=object)

    splits = list(GroupKFold(n_splits=args.folds).split(texts, y, groups))
    if single_fold_mode:
        if args.fold < 0 or args.fold >= args.folds:
            raise ValueError(f"--fold must be in [0, {args.folds - 1}]")
        _, val_idx = splits[args.fold]
        print(
            f"predict single-fold fold={args.fold} model={model_dirs[0]} rows={len(val_idx)}",
            flush=True,
        )
        fold_logits, _ = predict_logits_for_model(model_dirs[0], texts[val_idx].tolist(), args)
        y_eval = y[val_idx]
        logits_eval = fold_logits
    else:
        logits = np.zeros((len(texts), len(ACTION_CLASSES)), dtype=np.float32)
        for fold, (_, val_idx) in enumerate(splits):
            print(f"predict oof fold={fold} model={model_dirs[fold]} rows={len(val_idx)}", flush=True)
            logits[val_idx], _ = predict_logits_for_model(model_dirs[fold], texts[val_idx].tolist(), args)
        y_eval = y
        logits_eval = logits

    probs = softmax(logits_eval)
    order = np.argsort(-logits_eval, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    pred = top1
    correct = pred == y_eval
    margin = logits_eval[np.arange(len(logits_eval)), top1] - logits_eval[np.arange(len(logits_eval)), top2]
    confidence = probs[np.arange(len(probs)), top1]
    top1_labels = np.asarray(ACTION_CLASSES, dtype=object)[top1]

    macro_f1 = f1_score(y_eval, pred, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
    score_label = "val_macro_f1" if single_fold_mode else "oof_macro_f1"
    print(f"\n{score_label}={macro_f1:.6f}")
    print(f"accuracy={float(correct.mean()):.6f}")
    print(classification_report(y_eval, pred, target_names=ACTION_CLASSES, digits=4, zero_division=0))

    print("\n[overall margin/confidence]")
    for name, values in [("margin", margin), ("confidence", confidence)]:
        print(
            f"{name}: min={values.min():.4f} p01={np.quantile(values, 0.01):.4f} "
            f"p05={np.quantile(values, 0.05):.4f} p10={np.quantile(values, 0.10):.4f} "
            f"p25={np.quantile(values, 0.25):.4f} p50={np.quantile(values, 0.50):.4f} "
            f"p75={np.quantile(values, 0.75):.4f} p90={np.quantile(values, 0.90):.4f} "
            f"max={values.max():.4f}"
        )

    print_bucket_stats("margin", margin, correct, [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    print_bucket_stats("confidence", confidence, correct, [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7])

    print("\n[top1 class stats]")
    for label_id, label in enumerate(ACTION_CLASSES):
        mask = top1 == label_id
        if not mask.any():
            continue
        print(
            f"{label:18s} rows={int(mask.sum()):5d} acc={float(correct[mask].mean()):.4f} "
            f"margin_p50={np.quantile(margin[mask], 0.50):.4f} "
            f"conf_p50={np.quantile(confidence[mask], 0.50):.4f}"
        )

    search_labels = {"read_file", "grep_search", "list_directory", "glob_pattern"}
    search_mask = np.isin(top1_labels, list(search_labels))
    print("\n[search top1 subset]")
    print(f"rows={int(search_mask.sum())} acc={float(correct[search_mask].mean()):.4f}")
    print_bucket_stats("search_margin", margin[search_mask], correct[search_mask], [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0])
    print_bucket_stats("search_confidence", confidence[search_mask], correct[search_mask], [0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7])


if __name__ == "__main__":
    main()
