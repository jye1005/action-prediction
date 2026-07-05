import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.features import render_granite_text
from action_router.constants import ACTION_CLASSES


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


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    values = []
    bias_map = payload.get("bias", {})
    for idx in range(len(id2label)):
        values.append(float(bias_map.get(id2label[idx], 0.0)))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dir", default="./model/granite-311m-fold0")
    parser.add_argument("--output-path", default="./output/submission.csv")
    parser.add_argument(
        "--probs-path",
        default=None,
        help="Optional .npz path to save per-class logits and probabilities "
        "(columns ordered by ACTION_CLASSES) for ensembling/stacking.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--feature-mode", choices=["granite", "granite_v2", "granite_v3", "auto"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    class TextDataset:
        def __init__(self, texts, tokenizer, max_length):
            self.texts = texts
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            return self.tokenizer(
                self.texts[idx],
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

    model_dir = Path(args.model_dir)
    model_name_or_path = str(model_dir) if model_dir.exists() else args.model_dir
    feature_mode = args.feature_mode
    if feature_mode == "auto":
        meta_path = model_dir / "training_meta.json"
        if meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                feature_mode = json.load(f).get("feature_mode", "granite")
        else:
            feature_mode = "granite"
    print(f"feature_mode={feature_mode}")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    bias_values = load_logit_bias(model_dir, model.config.id2label)
    bias = torch.tensor(bias_values, dtype=torch.float32, device=device) if bias_values is not None else None

    samples = load_jsonl(Path(args.data_dir) / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [
        render_granite_text(sample, max_history_events=args.max_history_events, feature_mode=feature_mode)
        for sample in samples
    ]

    loader = DataLoader(
        TextDataset(texts, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    import numpy as np

    preds = []
    logit_chunks = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(**batch).logits
            logits = logits.float()
            if bias is not None:
                logits = logits + bias
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend([model.config.id2label[int(i)] for i in pred_ids])
            if args.probs_path:
                logit_chunks.append(logits.cpu().numpy())

    pred_map = dict(zip(ids, preds))
    fieldnames, rows = load_sample_submission(Path(args.data_dir) / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(args.output_path, fieldnames, rows)
    print(f"Saved {args.output_path} rows={len(rows)}")

    if args.probs_path:
        # Logits are in the model's own id order; reorder to canonical
        # ACTION_CLASSES so every model's columns line up for ensembling.
        all_logits = np.concatenate(logit_chunks, axis=0)
        label2id = {model.config.id2label[i]: i for i in range(all_logits.shape[1])}
        missing = [c for c in ACTION_CLASSES if c not in label2id]
        if missing:
            raise ValueError(f"model is missing action classes: {missing}")
        col_order = [label2id[name] for name in ACTION_CLASSES]
        ordered = all_logits[:, col_order].astype(np.float32)
        shifted = ordered - ordered.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probs = (exp / exp.sum(axis=1, keepdims=True)).astype(np.float32)
        out = Path(args.probs_path)
        os.makedirs(out.parent, exist_ok=True)
        np.savez(
            out,
            ids=np.array(ids),
            classes=np.array(ACTION_CLASSES),
            logits=ordered,
            probs=probs,
        )
        print(f"Saved probs to {args.probs_path} shape={probs.shape} (post-bias, ACTION_CLASSES order)")


if __name__ == "__main__":
    main()
