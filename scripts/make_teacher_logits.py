import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES, ID2LABEL, LABEL2ID
from action_router.features import render_granite_sample


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_base_model_name(model_dir, fallback):
    if fallback:
        return fallback
    meta_path = Path(model_dir) / "training_meta.json"
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("base_model", "")
    return ""


def is_lora_model(model_dir):
    return (Path(model_dir) / "adapter_config.json").exists()


def load_teacher(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=args.local_files_only)
    if is_lora_model(model_dir):
        from peft import PeftModel

        base_model_name = load_base_model_name(model_dir, args.base_model)
        if not base_model_name:
            raise ValueError("LoRA teacher에는 --base-model 또는 training_meta.json의 base_model이 필요합니다.")
        base_model = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=len(ACTION_CLASSES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            local_files_only=args.local_files_only,
            trust_remote_code=args.trust_remote_code,
            attn_implementation=args.attn_implementation,
        )
        model = PeftModel.from_pretrained(base_model, model_dir, local_files_only=args.local_files_only)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            local_files_only=args.local_files_only,
            attn_implementation=args.attn_implementation,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-path", default="./cache/teacher_train_logits.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    args = parser.parse_args()

    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    samples = load_jsonl(Path(args.data_dir) / f"{args.split}.jsonl")
    ids = [sample["id"] for sample in samples]
    labels = None
    if args.split == "train":
        label_map = load_labels(Path(args.data_dir) / "train_labels.csv")
        labels = np.asarray([LABEL2ID[label_map[sample_id]] for sample_id in ids], dtype=np.int64)
    texts = [render_granite_sample(sample, max_history_events=args.max_history_events) for sample in samples]

    tokenizer, model, device = load_teacher(args)

    class TextDataset:
        def __len__(self):
            return len(texts)

        def __getitem__(self, idx):
            return tokenizer(texts[idx], truncation=True, max_length=args.max_length, padding=False)

    loader = DataLoader(
        TextDataset(),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    logits = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch).logits.float().cpu().numpy()
            logits.append(out)
    logits = np.concatenate(logits, axis=0)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    payload = {"ids": np.asarray(ids, dtype=object), "logits": logits, "classes": np.asarray(ACTION_CLASSES, dtype=object)}
    if labels is not None:
        payload["labels"] = labels
    np.savez_compressed(args.output_path, **payload)
    print(f"saved {args.output_path} logits={logits.shape}")


if __name__ == "__main__":
    main()
