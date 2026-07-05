"""Verify int8 quantization on a fine-tuned sequence-classification router.

Purpose
-------
De-risk the 2-way (granite + bge) ensemble under the 1GB packaging limit by
measuring, for ONE model, both:

  1. accuracy retention: macro-F1 fp16 vs int8 on the exact same eval set, and
  2. on-disk size: original model dir vs saved int8 model dir.

The eval set is reconstructed from an existing baseline `.npz` dump (e.g.
`granite-fold0.npz`) by matching its `ids` back to the training JSONL, so the
int8 numbers are directly comparable to the fp baseline already stored in the
npz (same samples, same order, same y_true).

The int8 probabilities are re-dumped to `--out-npz` with the same schema
(ids, y_true, classes, logits, probs), so `blend_eval.py` can consume them and
you can measure the *blend* macro under int8 directly.

Quantization methods
--------------------
  --method bnb      bitsandbytes 8-bit (GPU; closest to T4 deploy). Default.
  --method dynamic  torch dynamic int8 on Linear layers (CPU-only, no extra deps).

Granite loads as ModernBertForSequenceClassification and needs
transformers==4.48.3; bge-m3 loads as XLMRobertaForSequenceClassification and
works on the default stack. Run each model in its matching environment.

Example
-------
    # granite (transformers==4.48.3 env)
    python scripts/verify_int8.py \
        --model-dir ./model/granite-311m-fold0 \
        --data-dir ./data \
        --baseline-npz ./granite-fold0.npz \
        --renderer granite \
        --method bnb \
        --out-npz ./granite-fold0-int8.npz

    # bge-m3
    python scripts/verify_int8.py \
        --model-dir ./model/bge-m3-fold0 \
        --data-dir ./data \
        --baseline-npz ./bge-m3-fold0.npz \
        --renderer granite \
        --method bnb \
        --out-npz ./bge-m3-fold0-int8.npz
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "src"))

from action_router.constants import ACTION_CLASSES  # noqa: E402
from action_router.features import render_granite_sample, render_sample  # noqa: E402

RENDERERS = {
    "granite": lambda s: render_granite_sample(s, max_history_events=12),
    "e5": lambda s: render_sample(s, max_history=8),
}


def macro_f1(y_true, y_pred, n_classes=len(ACTION_CLASSES)):
    """Unweighted mean per-class F1. Kept dependency-free (no sklearn)."""
    scores = []
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return float(np.mean(scores))


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dir_size_mb(path):
    total = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
    return total / 1e6


def build_eval_set(data_dir, baseline_npz, renderer):
    """Reconstruct texts for exactly the ids stored in the baseline npz."""
    base = np.load(baseline_npz, allow_pickle=True)
    ids = [str(i) for i in base["ids"]]
    y_true = base["y_true"].astype(np.int64)
    base_probs = base["probs"]

    samples = load_jsonl(Path(data_dir) / "train.jsonl")
    by_id = {str(s["id"]): s for s in samples}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise SystemExit(
            f"{len(missing)} ids from {baseline_npz} not found in train.jsonl "
            f"(first few: {missing[:3]}). Check --data-dir."
        )
    texts = [renderer(by_id[i]) for i in ids]
    return ids, texts, y_true, base_probs


def load_logit_bias(model_dir, id2label):
    path = Path(model_dir) / "logit_bias.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    bias_map = payload.get("bias", {})
    return [float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))]


def run_inference(model, tokenizer, texts, device, max_length, batch_size, bias, fp16):
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorWithPadding

    class DS:
        def __len__(self):
            return len(texts)

        def __getitem__(self, i):
            return tokenizer(texts[i], truncation=True, max_length=max_length, padding=False)

    loader = DataLoader(
        DS(),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )
    all_logits = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            use_amp = fp16 and device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(**batch).logits
            logits = logits.float()
            if bias is not None:
                logits = logits + bias
            all_logits.append(logits.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--baseline-npz", required=True,
                        help="Existing fp dump (ids/y_true/probs) to compare against and match eval set.")
    parser.add_argument("--renderer", choices=list(RENDERERS), default="granite")
    parser.add_argument("--method", choices=["bnb", "dynamic"], default="bnb")
    parser.add_argument("--out-npz", default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--skip-fp", action="store_true",
                        help="Trust baseline npz as the fp reference instead of recomputing it.")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    renderer = RENDERERS[args.renderer]
    ids, texts, y_true, base_probs = build_eval_set(args.data_dir, args.baseline_npz, renderer)
    print(f"eval set: {len(ids)} samples reconstructed from {args.baseline_npz}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- baseline (fp) reference ----
    base_macro = macro_f1(y_true, base_probs.argmax(1))
    print(f"[baseline npz] macro-F1 = {base_macro:.4f}")

    fp_macro = base_macro
    if not args.skip_fp:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True)
        model.to(device)
        bias_vals = load_logit_bias(args.model_dir, model.config.id2label)
        bias = torch.tensor(bias_vals, device=device) if bias_vals else None
        fp_logits = run_inference(model, tokenizer, texts, device, args.max_length, args.batch_size, bias, fp16=True)
        fp_macro = macro_f1(y_true, fp_logits.argmax(1))
        print(f"[fp16 recompute] macro-F1 = {fp_macro:.4f}  (drift vs npz: {fp_macro - base_macro:+.4f})")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- int8 ----
    orig_mb = dir_size_mb(args.model_dir)
    if args.method == "bnb":
        try:
            from transformers import BitsAndBytesConfig
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"bitsandbytes/transformers BitsAndBytesConfig unavailable: {e}. Try --method dynamic.")
        qconf = BitsAndBytesConfig(load_in_8bit=True)
        q_model = AutoModelForSequenceClassification.from_pretrained(
            args.model_dir, local_files_only=True, quantization_config=qconf, device_map="auto"
        )
        q_device = next(q_model.parameters()).device
        bias_vals = load_logit_bias(args.model_dir, q_model.config.id2label)
        bias = torch.tensor(bias_vals, device=q_device) if bias_vals else None
        int8_logits = run_inference(q_model, tokenizer, texts, q_device, args.max_length, args.batch_size, bias, fp16=False)
        int8_dir = Path(args.model_dir).parent / (Path(args.model_dir).name + "-int8")
        try:
            q_model.save_pretrained(int8_dir)
            tokenizer.save_pretrained(int8_dir)
            int8_mb = dir_size_mb(int8_dir)
        except Exception as e:  # noqa: BLE001
            int8_mb = float("nan")
            print(f"warning: could not serialize int8 model for size measurement: {e}")
    else:  # dynamic (CPU)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True)
        model.to("cpu").eval()
        q_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        bias_vals = load_logit_bias(args.model_dir, model.config.id2label)
        bias = torch.tensor(bias_vals) if bias_vals else None
        int8_logits = run_inference(q_model, tokenizer, texts, torch.device("cpu"), args.max_length, args.batch_size, bias, fp16=False)
        int8_dir = Path(args.model_dir).parent / (Path(args.model_dir).name + "-int8-dynamic")
        int8_dir.mkdir(parents=True, exist_ok=True)
        torch.save(q_model.state_dict(), int8_dir / "int8_state.pt")
        int8_mb = dir_size_mb(int8_dir)

    int8_macro = macro_f1(y_true, int8_logits.argmax(1))
    int8_probs = softmax(int8_logits)

    print("\n================ int8 verification ================")
    print(f"model            : {args.model_dir}")
    print(f"method           : {args.method}")
    print(f"macro-F1  fp      : {fp_macro:.4f}")
    print(f"macro-F1  int8    : {int8_macro:.4f}")
    print(f"macro-F1  delta   : {int8_macro - fp_macro:+.4f}")
    print(f"size  fp   (dir)  : {orig_mb:8.1f} MB")
    print(f"size  int8 (dir)  : {int8_mb:8.1f} MB")
    print("===================================================")

    if args.out_npz:
        np.savez(
            args.out_npz,
            ids=np.array(ids, dtype=object),
            y_true=y_true,
            classes=np.array(ACTION_CLASSES),
            logits=int8_logits.astype(np.float32),
            probs=int8_probs.astype(np.float32),
        )
        print(f"saved int8 dump -> {args.out_npz}")


if __name__ == "__main__":
    main()
