import csv
import json
import os
from pathlib import Path


def _safe_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _compact_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _budget_bucket(tokens):
    try:
        tokens = int(tokens)
    except (TypeError, ValueError):
        return "unknown"
    if tokens < 2_000:
        return "very_low"
    if tokens < 10_000:
        return "low"
    if tokens < 50_000:
        return "medium"
    return "high"


def _elapsed_bucket(seconds):
    try:
        seconds = int(seconds)
    except (TypeError, ValueError):
        return "unknown"
    if seconds < 120:
        return "early"
    if seconds < 900:
        return "mid"
    return "late"


def render_sample(sample, max_history=8):
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    recent_history = history[-max_history:]

    parts = [
        "task: predict next ai coding agent action",
        f"user_tier: {_safe_text(meta.get('user_tier'))}",
        f"language_pref: {_safe_text(meta.get('language_pref'))}",
        f"budget_bucket: {_budget_bucket(meta.get('budget_tokens_remaining'))}",
        f"turn_index: {_safe_text(meta.get('turn_index'))}",
        f"elapsed_bucket: {_elapsed_bucket(meta.get('elapsed_session_sec'))}",
        f"workspace_languages: {_compact_json(workspace.get('language_mix') or {})}",
        f"workspace_loc: {_safe_text(workspace.get('loc'))}",
        f"git_dirty: {_safe_text(workspace.get('git_dirty'))}",
        f"open_files: {_compact_json(workspace.get('open_files') or [])}",
        f"last_ci_status: {_safe_text(workspace.get('last_ci_status'))}",
    ]

    for item in recent_history:
        role = item.get("role", "")
        if role == "user":
            parts.append(f"history_user: {_safe_text(item.get('content'))}")
        elif role == "assistant_action":
            name = _safe_text(item.get("name"))
            args = _compact_json(item.get("args") or {})
            result = _safe_text(item.get("result_summary"))
            parts.append(f"history_action: {name} args={args} result={result}")
        else:
            parts.append(f"history_{role}: {_compact_json(item)}")

    parts.append(f"current_prompt: {_safe_text(sample.get('current_prompt'))}")
    return "\n".join(parts)


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


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    data_dir = Path("./data")
    model_dir = Path("./model/e5-small-router-val20")
    output_path = Path("./output/submission.csv")
    max_length = 512
    max_history = 8
    batch_size = 64

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [render_sample(sample, max_history=max_history) for sample in samples]

    class TextDataset:
        def __init__(self, items):
            self.items = items

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            return tokenizer(
                self.items[idx],
                truncation=True,
                max_length=max_length,
                padding=False,
            )

    loader = DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend([model.config.id2label[int(i)] for i in pred_ids])

    pred_map = dict(zip(ids, preds))
    fieldnames, rows = load_sample_submission(data_dir / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()

