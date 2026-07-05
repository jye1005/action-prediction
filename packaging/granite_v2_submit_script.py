"""Self-contained inference script for the granite-v2 single-model submission.

Fixes vs the old granite_submit_script.py (all three would tank the v2 model):
  1. renderer: uses the granite_v2 format (render_granite_sample_v2), embedded
     inline because the eval server has no src/ package.
  2. dtype: loads/runs in bf16 (NOT fp16 autocast). granite-embedding is
     ModernBert, which overflows to NaN in fp16 -> near-random output.
  3. model dir: points at the v2 checkpoint.

Packaging (submit.zip layout):
    submit.zip
      script.py                    <- this file, renamed to script.py
      requirements.txt             <- pin transformers==4.48.3, torch
      model/granite-311m-v2-fold0-1/   <- the fp16 checkpoint (~659MB)

The eval server provides ./data/test.jsonl and ./data/sample_submission.csv and
runs `python script.py`. Output goes to ./output/submission.csv.
"""

import csv
import json
import os
from pathlib import Path

# ---- edit this to match the checkpoint folder name inside the zip ----
MODEL_DIR = os.environ.get("MODEL_DIR", "./model/granite-311m-v2-fold0-1")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
MAX_HISTORY_EVENTS = 12
MAX_OPEN_FILES = 8
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))
# Proven-fast pattern (matches a passing submission): load default (fp32) + fp16
# autocast + default sdpa attention. fp32 master weights => no NaN; fp16 compute
# uses T4 tensor cores => fast. (bf16 is slow on T4; eager attn is slow.)
# dtype choice on T4 (int8 & fp32 are accurate AND have native T4 acceleration):
#   int8      -> accurate (0.7825), T4 int8 tensor cores = fast. needs bitsandbytes. DEFAULT.
#   fp32      -> accurate (0.7831), native FP32 cores. no bnb. medium speed.
#   bf16      -> accurate (0.7829) but SLOW on T4 (no bf16 tensor cores) -> timed out.
#   autocast  -> fp16 autocast: fast but BROKEN on ModernBert (macro ~0.30). do not use.
COMPUTE = os.environ.get("COMPUTE", "int8")  # "int8" | "fp32" | "bf16" | "autocast"
ATTN = os.environ.get("ATTN", "")            # "" = default(sdpa); or "eager"


# ======================= embedded granite_v2 renderer =======================
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


def _path_summary(path):
    text = _safe_text(path)
    base = os.path.basename(text)
    ext = os.path.splitext(base)[1]
    parent = os.path.basename(os.path.dirname(text))
    return "/".join(part for part in [parent, base] if part), ext


def _extract_arg_hints(args):
    if not isinstance(args, dict):
        return []
    hints = []
    for key in ["path", "file", "filename", "dir", "directory", "pattern", "query", "command", "cmd", "glob", "regex"]:
        if key in args:
            value = args.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                hints.append(f"{key}={_safe_text(value)}")
            else:
                hints.append(f"{key}={_compact_json(value)}")
    return hints


def render_granite_sample_v2(sample, max_history_events=12, max_open_files=8):
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    recent_history = history[-max_history_events:]

    open_files = workspace.get("open_files") or []
    language_mix = workspace.get("language_mix") or {}
    main_lang = ""
    if language_mix:
        main_lang = max(language_mix.items(), key=lambda item: item[1])[0]

    meta_text = " ".join(
        [
            f"tier={_safe_text(meta.get('user_tier'))}",
            f"pref={_safe_text(meta.get('language_pref'))}",
            f"turn={_safe_text(meta.get('turn_index'))}",
            f"budget={_budget_bucket(meta.get('budget_tokens_remaining'))}",
            f"elapsed={_elapsed_bucket(meta.get('elapsed_session_sec'))}",
            f"lang={main_lang}",
            f"ci={_safe_text(workspace.get('last_ci_status'))}",
            f"git={'dirty' if workspace.get('git_dirty') else 'clean'}",
            f"open={len(open_files)}",
            f"loc={_safe_text(workspace.get('loc'))}",
        ]
    )

    path_parts = []
    ext_parts = []
    for path in open_files[-max_open_files:]:
        short_path, ext = _path_summary(path)
        if short_path:
            path_parts.append(short_path)
        if ext:
            ext_parts.append(ext)

    action_names = []
    arg_parts = []
    hist_parts = []
    for item in recent_history:
        role = item.get("role", "")
        if role == "user":
            hist_parts.append(f"U: {_safe_text(item.get('content'))}")
        elif role == "assistant_action":
            name = _safe_text(item.get("name"))
            args = item.get("args") or {}
            result = _safe_text(item.get("result_summary"))
            action_names.append(name)
            arg_hints = _extract_arg_hints(args)
            if arg_hints:
                arg_parts.append(f"{name} " + " ".join(arg_hints))
            hist_parts.append(f"A[{name}] {_compact_json(args)} -> {result}")

    current_prompt = _safe_text(sample.get("current_prompt"))
    prompt_lower = current_prompt.lower()
    prompt_hints = []
    hint_groups = {
        "read": ["read", "open", "show", "view", "봐", "열어", "확인"],
        "grep": ["grep", "search", "find", "검색", "찾"],
        "list": ["list", "ls", "directory", "folder", "폴더", "목록"],
        "glob": ["glob", "*.py", "**/", "pattern", "패턴"],
    }
    for group, keywords in hint_groups.items():
        if any(keyword in prompt_lower for keyword in keywords):
            prompt_hints.append(group)

    return " ".join(
        [
            "[META]", meta_text,
            "[FILES]", " | ".join(path_parts),
            "[EXT]", " ".join(sorted(set(ext_parts))),
            "[ACT]", " > ".join(action_names[-6:]),
            "[ARGS]", " | ".join(arg_parts[-6:]),
            "[HIST]", " | ".join(hist_parts),
            "[HINT]", " ".join(prompt_hints),
            "[CUR]", current_prompt,
        ]
    )


# ============================== io helpers ==============================
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
    bias_map = payload.get("bias", {})
    return [float(bias_map.get(id2label[idx], 0.0)) for idx in range(len(id2label))]


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import time
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding

    # ModernBert triggers torch.compile; dynamo can't trace bnb int8 -> fall back to eager.
    try:
        import torch._dynamo
        torch._dynamo.config.suppress_errors = True
    except Exception:  # noqa: BLE001
        pass

    # Submission: server provides ./data. Local test: set DATA_DIR=../data.
    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    model_dir = Path(MODEL_DIR)
    output_path = Path("./output/submission.csv")
    print(f"batch={BATCH_SIZE} max_len={MAX_LENGTH} compute={COMPUTE} attn={ATTN or 'default'}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    attn = ATTN or "sdpa"
    if COMPUTE == "int8":
        # Load the fp16 checkpoint quantized to int8 (T4 int8 tensor cores = fast,
        # accuracy ~0.7825). Keep the classification head in fp (skip_modules) or
        # it collapses. No autocast; bnb handles compute dtype.
        from transformers import BitsAndBytesConfig
        qconf = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["classifier", "score", "pre_classifier", "head"],
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir, local_files_only=True, quantization_config=qconf,
            device_map="auto", reference_compile=False, attn_implementation=attn,
        )
        device = next(model.parameters()).device
    else:
        load_kwargs = {"local_files_only": True}
        if ATTN:
            load_kwargs["attn_implementation"] = ATTN
        if COMPUTE == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, **load_kwargs)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
    model.eval()

    bias_values = load_logit_bias(model_dir, model.config.id2label)
    bias = torch.tensor(bias_values, dtype=torch.float32, device=device) if bias_values is not None else None

    samples = load_jsonl(data_dir / "test.jsonl")
    ids = [sample["id"] for sample in samples]
    texts = [render_granite_sample_v2(s, max_history_events=MAX_HISTORY_EVENTS, max_open_files=MAX_OPEN_FILES) for s in samples]

    class TextDataset:
        def __len__(self):
            return len(texts)

        def __getitem__(self, idx):
            return tokenizer(texts[idx], truncation=True, max_length=MAX_LENGTH, padding=False)

    loader = DataLoader(
        TextDataset(),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    preds = []
    use_autocast = COMPUTE == "autocast"
    t0 = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.amp.autocast("cuda", enabled=use_autocast and device.type == "cuda"):
                logits = model(**batch).logits
            logits = logits.float()
            if bias is not None:
                logits = logits + bias
            pred_ids = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            preds.extend([model.config.id2label[int(i)] for i in pred_ids])
    print(f"inference: {len(preds)} rows in {time.time() - t0:.1f}s", flush=True)

    pred_map = dict(zip(ids, preds))
    fieldnames, rows = load_sample_submission(data_dir / "sample_submission.csv")
    for row in rows:
        row["action"] = pred_map[row["id"]]
    save_submission(output_path, fieldnames, rows)
    print(f"Saved {output_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
