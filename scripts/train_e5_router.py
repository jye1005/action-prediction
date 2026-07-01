import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, ID2LABEL, LABEL2ID
from action_router.features import render_sample


class ActionDataset:
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = int(self.labels[idx])
        return encoded


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def build_xy(data_dir, max_history, max_samples=None, seed=42):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    texts = []
    y = []
    for sample in samples:
        action = labels[sample["id"]]
        texts.append(render_sample(sample, max_history=max_history))
        y.append(LABEL2ID[action])
    y = np.array(y, dtype=np.int64)
    if max_samples and max_samples < len(texts):
        rng = np.random.default_rng(seed)
        chosen = []
        per_class = max(1, max_samples // len(ACTION_CLASSES))
        for label_id in range(len(ACTION_CLASSES)):
            idx = np.flatnonzero(y == label_id)
            take = min(per_class, len(idx))
            chosen.extend(rng.choice(idx, size=take, replace=False).tolist())
        if len(chosen) < max_samples:
            remaining = np.setdiff1d(np.arange(len(texts)), np.array(chosen), assume_unique=False)
            take = min(max_samples - len(chosen), len(remaining))
            chosen.extend(rng.choice(remaining, size=take, replace=False).tolist())
        chosen = np.array(chosen[:max_samples])
        rng.shuffle(chosen)
        texts = [texts[i] for i in chosen]
        y = y[chosen]
    return texts, y


def make_class_weights(y):
    counts = Counter(int(v) for v in y)
    weights = []
    total = len(y)
    for i in range(len(ACTION_CLASSES)):
        # Square-root inverse frequency is less brittle than full inverse freq.
        weights.append((total / max(counts[i], 1)) ** 0.5)
    weights = np.array(weights, dtype=np.float32)
    return weights / weights.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-small")
    parser.add_argument("--output-dir", default="./model/e5-small-router")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        get_linear_schedule_with_warmup,
        set_seed,
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    texts, y = build_xy(args.data_dir, args.max_history, max_samples=args.max_samples, seed=args.seed)
    train_texts, val_texts, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=0.15,
        random_state=args.seed,
        stratify=y,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(ACTION_CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)

    train_ds = ActionDataset(train_texts, y_train, tokenizer, args.max_length)
    val_ds = ActionDataset(val_texts, y_val, tokenizer, args.max_length)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    class_weights = torch.tensor(make_class_weights(y_train), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.06)),
        num_training_steps=total_steps,
    )

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item())
            if step % 200 == 0:
                print(f"epoch={epoch} step={step}/{len(train_loader)} loss={running_loss / step:.4f}")

        model.eval()
        preds = []
        gold = []
        with torch.no_grad():
            for batch in val_loader:
                labels = batch.pop("labels").numpy().tolist()
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**batch).logits
                preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
                gold.extend(labels)
        macro_f1 = f1_score(gold, preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
        print(f"epoch={epoch} val_macro_f1={macro_f1:.5f}")
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            with open(Path(args.output_dir) / "training_meta.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "base_model": args.model_name,
                        "best_val_macro_f1": best_f1,
                        "max_length": args.max_length,
                        "max_history": args.max_history,
                        "action_classes": ACTION_CLASSES,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"saved best model to {args.output_dir}")


if __name__ == "__main__":
    main()
