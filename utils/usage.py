"""Running tally of Claude token usage.

Every API call is made from a thread in this process (evolution.parallel.run_parallel), and the
process pool only ever runs fitness code, so one module-level counter under a lock
sees every call.
"""
import threading

FIELDS = (
    "calls",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)

_lock = threading.Lock()
_totals = dict.fromkeys(FIELDS, 0)


def record(response):
    """Add one API response to the tally."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    details = getattr(usage, "output_tokens_details", None)
    counts = {
        "calls": 1,
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "thinking_tokens": getattr(details, "thinking_tokens", 0) or 0,
        "cache_read_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
        "cache_creation_tokens": getattr(usage, "cache_creation_input_tokens", 0) or 0,
    }
    with _lock:
        for field, value in counts.items():
            _totals[field] += value


def totals():
    with _lock:
        return dict(_totals)


def delta(previous):
    """Usage accumulated since the `previous` snapshot from totals()."""
    current = totals()
    return {field: current[field] - previous.get(field, 0) for field in FIELDS}


def reset():
    with _lock:
        for field in FIELDS:
            _totals[field] = 0


def describe(stats):
    """One-line human summary. Text tokens are output minus thinking."""
    text_tokens = stats["output_tokens"] - stats["thinking_tokens"]
    return (
        f"{stats['calls']} call(s), "
        f"in={stats['input_tokens']}, "
        f"out={stats['output_tokens']} "
        f"(thinking={stats['thinking_tokens']}, text={text_tokens})"
    )


def as_metrics(stats, prefix):
    """Flatten to wandb-friendly keys, e.g. tokens/generation/input_tokens."""
    metrics = {f"{prefix}/{field}": stats[field] for field in FIELDS}
    metrics[f"{prefix}/text_tokens"] = stats["output_tokens"] - stats["thinking_tokens"]
    return metrics
