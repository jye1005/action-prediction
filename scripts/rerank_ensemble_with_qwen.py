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
from action_router.features import render_granite_sample, render_sample


SEARCH_LABELS = {"read_file", "grep_search", "list_directory", "glob_pattern"}


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


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


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
            return tokenizer(texts[idx], truncation=True, max_length=args.max_length, padding=False)

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

    return logits


def ensemble_logits(model_dirs, texts, args, weights=None):
    if weights is None:
        weights = np.full(len(model_dirs), 1.0 / len(model_dirs), dtype=np.float32)
    result = None
    for model_dir, weight in zip(model_dirs, weights):
        print(f"predict ensemble model={model_dir} weight={weight:.4f} rows={len(texts)}", flush=True)
        logits = predict_logits_for_model(model_dir, texts, args)
        result = logits * weight if result is None else result + logits * weight
    return result


def build_oof_logits(model_dirs, samples, y, args):
    if len(model_dirs) != args.folds:
        raise ValueError("--score-oof expects one model dir per fold.")
    ids = [sample["id"] for sample in samples]
    groups = np.asarray([session_group(sample_id) for sample_id in ids], dtype=object)
    texts = np.asarray(render_texts(samples, args.feature_mode, args.max_history, args.max_history_events), dtype=object)
    splits = list(GroupKFold(n_splits=args.folds).split(texts, y, groups))
    logits = np.zeros((len(texts), len(ACTION_CLASSES)), dtype=np.float32)
    for fold, (_, val_idx) in enumerate(splits):
        print(f"predict oof fold={fold} model={model_dirs[fold]} rows={len(val_idx)}", flush=True)
        logits[val_idx] = predict_logits_for_model(model_dirs[fold], texts[val_idx].tolist(), args)
    return texts.tolist(), logits


def select_rerank_indices(logits, args):
    probs = softmax(logits)
    order = np.argsort(-logits, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    margins = logits[np.arange(len(logits)), top1] - logits[np.arange(len(logits)), top2]
    conf = probs[np.arange(len(logits)), top1]
    top_labels = np.asarray(ACTION_CLASSES, dtype=object)[top1]

    selected = (margins <= args.margin_threshold) | (conf <= args.confidence_threshold)
    if args.rerank_search_only:
        selected = selected | np.isin(top_labels, list(SEARCH_LABELS))
    if args.max_rerank > 0:
        # Prioritize uncertain rows first.
        uncertainty = margins + conf
        candidate_idx = np.flatnonzero(selected)
        keep = candidate_idx[np.argsort(uncertainty[candidate_idx])[: args.max_rerank]]
        mask = np.zeros(len(logits), dtype=bool)
        mask[keep] = True
        selected = mask
    return np.flatnonzero(selected), order


def make_prompt(text, candidates):
    candidate_lines = "\n".join(f"- {label}" for label in candidates)
    return (
        "You are predicting the next action of an AI coding agent.\n"
        "Choose exactly one label from the candidates.\n"
        "Answer with only the label string, no explanation.\n\n"
        f"Candidates:\n{candidate_lines}\n\n"
        f"Session:\n{text}\n\n"
        "Answer:"
    )


def score_candidate(llm_model, llm_tokenizer, device, prompt, label, max_prompt_length):
    import torch

    prompt_ids = llm_tokenizer(prompt, truncation=True, max_length=max_prompt_length, return_tensors="pt").input_ids[0]
    label_ids = llm_tokenizer(" " + label, add_special_tokens=False, return_tensors="pt").input_ids[0]
    input_ids = torch.cat([prompt_ids, label_ids], dim=0).unsqueeze(0).to(device)
    labels = torch.full_like(input_ids, -100)
    labels[0, -len(label_ids) :] = input_ids[0, -len(label_ids) :]

    with torch.no_grad():
        out = llm_model(input_ids=input_ids, labels=labels)
    return -float(out.loss.item())


def load_llm(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model, trust_remote_code=args.trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        args.llm_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        trust_remote_code=args.trust_remote_code,
    )
    ensure_pad_token(tokenizer, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def rerank_with_llm(texts, logits, order, indices, args):
    llm_tokenizer, llm_model, device = load_llm(args)
    reranked = np.argmax(logits, axis=1).copy()
    changed = 0
    for count, idx in enumerate(indices, start=1):
        top_labels = [ACTION_CLASSES[class_id] for class_id in order[idx, : args.top_k]]
        if args.force_search_candidates and any(label in SEARCH_LABELS for label in top_labels):
            candidates = list(dict.fromkeys(top_labels + sorted(SEARCH_LABELS)))
        else:
            candidates = top_labels
        prompt = make_prompt(texts[idx], candidates)
        scores = [
            score_candidate(llm_model, llm_tokenizer, device, prompt, label, args.max_prompt_length)
            for label in candidates
        ]
        best_label = candidates[int(np.argmax(scores))]
        new_pred = ACTION_CLASSES.index(best_label)
        if new_pred != reranked[idx]:
            changed += 1
        reranked[idx] = new_pred
        if count % 50 == 0:
            print(f"reranked={count}/{len(indices)} changed={changed}", flush=True)
    print(f"rerank_done rows={len(indices)} changed={changed}", flush=True)
    return reranked


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--model-dirs", required=True)
    parser.add_argument("--output-path", default="./output/submission_rerank_qwen.csv")
    parser.add_argument("--mode", choices=["test", "oof"], default="test")
    parser.add_argument("--feature-mode", choices=["granite", "sample"], default="granite")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-history", type=int, default=8)
    parser.add_argument("--max-history-events", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--weights", default="")
    parser.add_argument("--use-logit-bias", action="store_true", default=True)
    parser.add_argument("--no-logit-bias", dest="use_logit_bias", action="store_false")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--margin-threshold", type=float, default=0.7)
    parser.add_argument("--confidence-threshold", type=float, default=0.55)
    parser.add_argument("--max-rerank", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--rerank-search-only", action="store_true", default=True)
    parser.add_argument("--force-search-candidates", action="store_true", default=True)
    parser.add_argument("--max-prompt-length", type=int, default=1536)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dirs = split_csv(args.model_dirs)
    weights = None
    if args.weights:
        weights = np.asarray([float(value) for value in split_csv(args.weights)], dtype=np.float32)
        weights = weights / weights.sum()

    if args.mode == "oof":
        samples = load_jsonl(Path(args.data_dir) / "train.jsonl")
        labels = load_labels(Path(args.data_dir) / "train_labels.csv")
        y = np.asarray([LABEL2ID[labels[sample["id"]]] for sample in samples], dtype=np.int64)
        texts, logits = build_oof_logits(model_dirs, samples, y, args)
        base_preds = np.argmax(logits, axis=1)
        base_f1 = f1_score(y, base_preds, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
        print(f"base_oof_macro_f1={base_f1:.6f}")
        indices, order = select_rerank_indices(logits, args)
        print(f"selected_for_rerank={len(indices)} / {len(texts)}")
        reranked = rerank_with_llm(texts, logits, order, indices, args)
        rerank_f1 = f1_score(y, reranked, labels=list(range(len(ACTION_CLASSES))), average="macro", zero_division=0)
        print(f"reranked_oof_macro_f1={rerank_f1:.6f}")
        print(classification_report(y, reranked, target_names=ACTION_CLASSES, digits=4, zero_division=0))
        return

    samples = load_jsonl(Path(args.data_dir) / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = render_texts(samples, args.feature_mode, args.max_history, args.max_history_events)
    logits = ensemble_logits(model_dirs, texts, args, weights)
    indices, order = select_rerank_indices(logits, args)
    print(f"selected_for_rerank={len(indices)} / {len(texts)}")
    pred_ids = rerank_with_llm(texts, logits, order, indices, args)
    preds = [ACTION_CLASSES[int(idx)] for idx in pred_ids]
    pred_map = dict(zip(ids, preds))

    fieldnames, rows = load_sample_submission(Path(args.data_dir) / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(args.output_path, fieldnames, rows)
    print(f"Saved {args.output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
