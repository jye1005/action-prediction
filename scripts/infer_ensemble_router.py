import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_sample, render_sample


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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


def ensure_pad_token(tokenizer, model=None):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return np.asarray([float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))], dtype=np.float32)


def render_texts(samples, feature_mode, max_history, max_history_events):
    if feature_mode == "sample":
        return [render_sample(sample, max_history=max_history) for sample in samples]
    if feature_mode == "granite":
        return [render_granite_sample(sample, max_history_events=max_history_events) for sample in samples]
    raise ValueError(f"unknown feature_mode: {feature_mode}")


def predict_logits_for_model(model_dir, texts, args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=args.local_files_only)
    model_kwargs = {"local_files_only": args.local_files_only}
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, **model_kwargs)
    ensure_pad_token(tokenizer, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    class TextDataset:
        def __len__(self):
            return len(texts)

        def __getitem__(self, idx):
            return tokenizer(
                texts[idx],
                truncation=True,
                max_length=args.max_length,
                padding=False,
            )

    loader = DataLoader(
        TextDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                logits = model(**batch).logits
            chunks.append(logits.float().cpu().numpy())
    logits = np.concatenate(chunks, axis=0)

    if args.use_logit_bias:
        bias = load_logit_bias(model_dir, model.config.id2label)
        if bias is not None:
            logits = logits + bias.reshape(1, -1)

    return logits, model.config.id2label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dirs", required=True, help="Comma-separated fold model dirs.")
    parser.add_argument("--output-path", default="./output/submission_ensemble.csv")
    parser.add_argument("--feature-mode", choices=["granite", "sample"], default="granite")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--weights", default="", help="Optional comma-separated model weights.")
    parser.add_argument("--use-logit-bias", action="store_true", default=True)
    parser.add_argument("--no-logit-bias", dest="use_logit_bias", action="store_false")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--amp", action="store_true", default=True)
    args = parser.parse_args()

    model_dirs = split_csv(args.model_dirs)
    if not model_dirs:
        raise ValueError("--model-dirs must contain at least one directory.")

    if args.weights:
        weights = np.asarray([float(value) for value in split_csv(args.weights)], dtype=np.float32)
        if len(weights) != len(model_dirs):
            raise ValueError("--weights length must match --model-dirs length.")
        weights = weights / weights.sum()
    else:
        weights = np.full(len(model_dirs), 1.0 / len(model_dirs), dtype=np.float32)

    samples = load_jsonl(Path(args.data_dir) / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = render_texts(samples, args.feature_mode, args.max_history, args.max_history_events)

    ensemble_logits = None
    id2label = None
    for model_dir, weight in zip(model_dirs, weights):
        print(f"predict model={model_dir} weight={weight:.4f}", flush=True)
        logits, current_id2label = predict_logits_for_model(model_dir, texts, args)
        if ensemble_logits is None:
            ensemble_logits = logits * weight
            id2label = current_id2label
        else:
            ensemble_logits += logits * weight

    pred_ids = np.argmax(ensemble_logits, axis=1).tolist()
    preds = [id2label[int(idx)] for idx in pred_ids]
    pred_map = dict(zip(ids, preds))

    fieldnames, rows = load_sample_submission(Path(args.data_dir) / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(args.output_path, fieldnames, rows)
    print(f"Saved {args.output_path} rows={len(rows)} models={len(model_dirs)}")


if __name__ == "__main__":
    main()
