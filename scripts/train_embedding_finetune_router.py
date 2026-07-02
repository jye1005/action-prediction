import argparse
import csv
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, train_test_split

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, ID2LABEL, LABEL2ID
from action_router.features import render_sample, session_group


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


def build_data(data_dir, max_history):
    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    labels = load_labels(Path(data_dir) / "train_labels.csv")
    texts = []
    y = []
    groups = []
    for sample in samples:
        sample_id = sample["id"]
        texts.append(render_sample(sample, max_history=max_history))
        y.append(LABEL2ID[labels[sample_id]])
        groups.append(session_group(sample_id))
    return np.array(texts, dtype=object), np.array(y, dtype=np.int64), np.array(groups, dtype=object)


def class_weights(y):
    counts = Counter(int(v) for v in y)
    total = len(y)
    weights = []
    for label_id in range(len(ACTION_CLASSES)):
        # Sqrt inverse frequency is usually less brittle than full inverse frequency.
        weights.append((total / max(counts[label_id], 1)) ** 0.5)
    weights = np.array(weights, dtype=np.float32)
    return weights / weights.mean()


def split_train_val(texts, y, groups, args):
    if args.split_mode == "group":
        splits = list(GroupKFold(n_splits=args.n_splits).split(texts, y, groups))
        train_idx, val_idx = splits[args.fold]
        return texts[train_idx].tolist(), texts[val_idx].tolist(), y[train_idx], y[val_idx]
    train_texts, val_texts, y_train, y_val = train_test_split(
        texts.tolist(),
        y,
        test_size=args.val_size,
        stratify=y,
        random_state=args.seed,
    )
    return train_texts, val_texts, y_train, y_val


def evaluate(model, loader, device, use_amp, amp_dtype):
    import torch

    model.eval()
    preds = []
    gold = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").numpy().tolist()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda", dtype=amp_dtype):
                logits = model(**batch).logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy().tolist())
            gold.extend(labels)
    return f1_score(gold, preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)


def ensure_pad_token(tokenizer, model=None):
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    if model is not None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-name", default="intfloat/multilingual-e5-base")
    parser.add_argument("--output-dir", default="./model/e5-base-finetune-router")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-mode", choices=["group", "stratified"], default="group")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="")
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
    texts, y, groups = build_data(args.data_dir, args.max_history)
    train_texts, val_texts, y_train, y_val = split_train_val(texts, y, groups, args)
    print(f"train={len(train_texts)} val={len(val_texts)} model={args.model_name} split={args.split_mode}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_pad_token(tokenizer)
    model_kwargs = {
        "num_labels": len(ACTION_CLASSES),
        "id2label": ID2LABEL,
        "label2id": LABEL2ID,
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, **model_kwargs)
    ensure_pad_token(tokenizer, model)
    model.to(device)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        ActionDataset(train_texts, y_train, tokenizer, args.max_length),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
    )
    val_loader = DataLoader(
        ActionDataset(val_texts, y_val, tokenizer, args.max_length),
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
    )

    weights = torch.tensor(class_weights(y_train), dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    update_steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = update_steps_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * args.warmup_ratio)),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16 and device.type == "cuda")
    use_amp = (args.fp16 or args.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16

    best_f1 = -1.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        update_step = 0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(**batch).logits
                loss = criterion(logits, labels) / args.grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.item()) * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1
                if update_step % 50 == 0:
                    print(
                        f"epoch={epoch} update={update_step}/{update_steps_per_epoch} "
                        f"loss={running_loss / step:.4f}"
                    )

        macro_f1 = evaluate(model, val_loader, device, use_amp, amp_dtype)
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
                        "split_mode": args.split_mode,
                        "action_classes": ACTION_CLASSES,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"saved best model to {args.output_dir}")


if __name__ == "__main__":
    main()
