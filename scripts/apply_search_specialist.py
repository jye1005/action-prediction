import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, LABEL2ID
from action_router.features import render_granite_sample
from infer_ensemble_router import predict_logits_for_model


SEARCH_CLASSES = ["read_file", "grep_search", "list_directory", "glob_pattern"]
SEARCH_SET = set(SEARCH_CLASSES)
SEARCH_TO_GLOBAL = np.asarray([ACTION_CLASSES.index(label) for label in SEARCH_CLASSES], dtype=np.int64)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_sample_submission(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def save_submission(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def session_group(sample_id):
    return str(sample_id).split("-step_", 1)[0]


def render_texts(samples, max_history_events):
    return [render_granite_sample(sample, max_history_events=max_history_events) for sample in samples]


def build_oof_logits(model_dirs, samples, y, args):
    ids = [sample["id"] for sample in samples]
    texts = np.asarray(render_texts(samples, args.max_history_events), dtype=object)
    groups = np.asarray([session_group(sample_id) for sample_id in ids], dtype=object)
    splits = list(GroupKFold(n_splits=args.folds).split(texts, y, groups))
    logits = np.zeros((len(texts), len(ACTION_CLASSES)), dtype=np.float32)
    for fold, (_, val_idx) in enumerate(splits):
        print(f"predict base oof fold={fold} model={model_dirs[fold]} rows={len(val_idx)}", flush=True)
        logits[val_idx], _ = predict_logits_for_model(model_dirs[fold], texts[val_idx].tolist(), args)
    return texts.tolist(), logits, splits


def build_test_logits(model_dirs, samples, args):
    texts = render_texts(samples, args.max_history_events)
    weights = np.full(len(model_dirs), 1.0 / len(model_dirs), dtype=np.float32)
    logits = None
    for model_dir, weight in zip(model_dirs, weights):
        print(f"predict base test model={model_dir} weight={weight:.4f}", flush=True)
        current, _ = predict_logits_for_model(model_dir, texts, args)
        logits = current * weight if logits is None else logits + current * weight
    return texts, logits


def predict_search_logits(model_dir, texts, args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=args.local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    class TextDataset:
        def __len__(self):
            return len(texts)

        def __getitem__(self, idx):
            return tokenizer(texts[idx], truncation=True, max_length=args.max_length, padding=False)

    loader = DataLoader(
        TextDataset(),
        batch_size=args.search_batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(**batch).logits
            chunks.append(logits.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def apply_specialist(base_logits, texts, specialist_dirs, args, splits=None):
    base_pred = np.argmax(base_logits, axis=1)
    base_labels = np.asarray(ACTION_CLASSES, dtype=object)[base_pred]
    selected = np.isin(base_labels, list(SEARCH_SET))

    if args.margin_threshold > 0:
        order = np.argsort(-base_logits, axis=1)
        margin = base_logits[np.arange(len(base_logits)), order[:, 0]] - base_logits[np.arange(len(base_logits)), order[:, 1]]
        selected = selected | (margin <= args.margin_threshold)

    selected_idx = np.flatnonzero(selected)
    final_pred = base_pred.copy()
    print(f"specialist_selected={len(selected_idx)} / {len(base_logits)}", flush=True)
    if len(selected_idx) == 0:
        return final_pred

    if splits is None:
        search_logits = None
        weights = np.full(len(specialist_dirs), 1.0 / len(specialist_dirs), dtype=np.float32)
        selected_texts = [texts[idx] for idx in selected_idx]
        for model_dir, weight in zip(specialist_dirs, weights):
            print(f"predict search specialist model={model_dir} weight={weight:.4f}", flush=True)
            current = predict_search_logits(model_dir, selected_texts, args)
            search_logits = current * weight if search_logits is None else search_logits + current * weight
        final_pred[selected_idx] = SEARCH_TO_GLOBAL[np.argmax(search_logits, axis=1)]
        return final_pred

    if len(specialist_dirs) != args.folds:
        raise ValueError("OOF mode expects one search specialist dir per fold.")
    for fold, (_, val_idx) in enumerate(splits):
        fold_idx = np.intersect1d(selected_idx, val_idx, assume_unique=False)
        if len(fold_idx) == 0:
            continue
        print(f"predict search oof fold={fold} model={specialist_dirs[fold]} rows={len(fold_idx)}", flush=True)
        fold_texts = [texts[idx] for idx in fold_idx]
        search_logits = predict_search_logits(specialist_dirs[fold], fold_texts, args)
        final_pred[fold_idx] = SEARCH_TO_GLOBAL[np.argmax(search_logits, axis=1)]
    return final_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--base-model-dirs", required=True)
    parser.add_argument("--specialist-dirs", required=True)
    parser.add_argument("--mode", choices=["oof", "test"], default="oof")
    parser.add_argument("--output-path", default="./output/submission_search_specialist.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--search-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--use-logit-bias", action="store_true", default=True)
    parser.add_argument("--no-logit-bias", dest="use_logit_bias", action="store_false")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    base_dirs = split_csv(args.base_model_dirs)
    specialist_dirs = split_csv(args.specialist_dirs)

    if args.mode == "oof":
        samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
        labels = load_labels(Path(args.data_dir) / "train_labels.csv")
        y = np.asarray([LABEL2ID[labels[sample["id"]]] for sample in samples], dtype=np.int64)
        texts, base_logits, splits = build_oof_logits(base_dirs, samples, y, args)
        base_pred = np.argmax(base_logits, axis=1)
        base_f1 = f1_score(y, base_pred, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
        print(f"base_oof_macro_f1={base_f1:.6f}")
        final_pred = apply_specialist(base_logits, texts, specialist_dirs, args, splits=splits)
        final_f1 = f1_score(y, final_pred, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
        print(f"search_specialist_oof_macro_f1={final_f1:.6f}")
        print(classification_report(y, final_pred, target_names=ACTION_CLASSES, digits=4, zero_division=0))
        return

    samples = load_jsonl(Path(args.data_dir) / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts, base_logits = build_test_logits(base_dirs, samples, args)
    final_pred = apply_specialist(base_logits, texts, specialist_dirs, args)
    preds = [ACTION_CLASSES[int(idx)] for idx in final_pred]
    pred_map = dict(zip(ids, preds))

    fieldnames, rows = load_sample_submission(Path(args.data_dir) / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(args.output_path, fieldnames, rows)
    print(f"Saved {args.output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
