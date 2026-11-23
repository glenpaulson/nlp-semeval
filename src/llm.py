"""Cached, retrying, thread-safe LLM client on litellm, with running token and call totals."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")  # use litellm's bundled price map, no fetch at import

import litellm  # noqa: E402

litellm.suppress_debug_info = True

CACHE_DIR = REPO_ROOT / "cache" / "llm"
TIMEOUT_S = 60
MAX_TRIES = 5
LONG_INPUT_TOKENS = 100_000
RETRYABLE_ERRORS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.BadGatewayError,
)
RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}

_lock = threading.Lock()
_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0, "cache_hits": 0}


class LLMError(RuntimeError):
    """A response that is not usable and must not be cached, such as empty text."""


def resolve_model(model: str | None = None) -> str:
    """Return model, or LLM_MODEL from the environment or .env when model is None."""
    model = model or os.environ.get("LLM_MODEL")
    if not model:
        raise ValueError("model is None and LLM_MODEL is not set; add LLM_MODEL to .env (see .env.example)")
    return model


def cache_key(request: dict[str, Any]) -> str:
    """Return the sha256 of the canonical JSON of the request."""
    blob = json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def usage_summary() -> dict[str, int]:
    """Return running totals of input/output tokens, successful API calls and cache hits."""
    with _lock:
        return dict(_usage)


def _add_usage(**deltas: int) -> None:
    with _lock:
        for name, n in deltas.items():
            _usage[name] += n


def _read_cache(path: Path) -> dict[str, Any] | None:
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_cache(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)  # atomic on POSIX and Windows, so readers never see a partial file
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _is_retryable(error: Exception) -> bool:
    return isinstance(error, RETRYABLE_ERRORS) or getattr(error, "status_code", None) in RETRYABLE_STATUS


def _retry_after(error: Exception) -> float:
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    try:
        return float(headers.get("retry-after", 0))
    except (AttributeError, TypeError, ValueError):
        return 0.0


def _call_api(request: dict[str, Any]) -> Any:
    sampling = {} if request["temperature"] is None else {"temperature": request["temperature"]}
    for attempt in range(1, MAX_TRIES + 1):
        try:
            return litellm.completion(
                model=request["model"],
                messages=request["messages"],
                max_tokens=request["max_tokens"],
                timeout=TIMEOUT_S,
                num_retries=0,
                **sampling,
                **request["kwargs"],
            )
        except Exception as error:
            if attempt == MAX_TRIES or not _is_retryable(error):
                raise
            backoff = min(2.0 * 2 ** (attempt - 1), 60.0) + random.uniform(0, 1)
            time.sleep(max(backoff, _retry_after(error)))
    raise AssertionError("unreachable")


def complete(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 512,
    refresh: bool = False,
    **kwargs: Any,
) -> str:
    """Return the response text for messages; temperature=None omits it; refresh=True skips the cache read."""
    if "api_key" in kwargs:
        raise ValueError("pass credentials through .env, not api_key=, so they never reach the cache")
    request = {
        "model": resolve_model(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "kwargs": kwargs,
    }
    key = cache_key(request)
    path = CACHE_DIR / f"{key}.json"
    if not refresh:
        cached = _read_cache(path)
        if cached is not None:
            _add_usage(cache_hits=1)
            return cached["response"]

    response = _call_api(request)
    choice = response.choices[0]
    text = choice.message.content or ""
    finish_reason = choice.finish_reason
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    _add_usage(input_tokens=input_tokens, output_tokens=output_tokens, api_calls=1)

    if input_tokens > LONG_INPUT_TOKENS:
        warnings.warn(
            f"{request['model']}: input was {input_tokens:,} tokens (> {LONG_INPUT_TOKENS:,}); "
            "Haiku 5.5 bills 5x above that",
            stacklevel=2,
        )
    if finish_reason == "content_filter":
        raise LLMError(f"{request['model']} declined the request (finish_reason=content_filter)")
    if not text.strip():
        raise LLMError(f"empty response (finish_reason={finish_reason}); try a larger max_tokens")
    if finish_reason == "length":
        warnings.warn(f"response truncated at max_tokens={max_tokens}", stacklevel=2)

    _write_cache(
        path,
        {
            "key": key,
            "request": request,
            "response": text,
            "finish_reason": finish_reason,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )
    return text


def complete_many(
    list_of_messages: list[list[dict[str, Any]]],
    concurrency: int = 8,
    return_exceptions: bool = False,
    **kwargs: Any,
) -> list[Any]:
    """Run complete() in a thread pool, results in input order; return_exceptions=True puts failures in place."""
    results: list[Any] = [None] * len(list_of_messages)
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(complete, messages, **kwargs): i for i, messages in enumerate(list_of_messages)}
        try:
            for future in tqdm(as_completed(futures), total=len(futures), desc="LLM"):
                error = future.exception() if return_exceptions else None
                results[futures[future]] = error if error is not None else future.result()
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)  # finished results stay cached, so a rerun resumes
            raise
    return results
