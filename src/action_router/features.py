import json


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
    """Render one JSONL sample into a compact text input for a classifier.

    The public baseline only uses current_prompt. This representation adds the
    recent agent trajectory and workspace state because next actions often
    depend on what was just read, edited, or tested.
    """
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

