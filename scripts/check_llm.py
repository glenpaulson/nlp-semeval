"""Smoke-test src/llm.py: one tiny call made twice, the second must be a cache hit."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import complete, resolve_model, usage_summary  # noqa: E402

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("ANTHROPIC_API_KEY is not set: add it to .env (see .env.example) and re-run.")
try:
    model = resolve_model()
except ValueError as error:
    sys.exit(str(error))

messages = [{"role": "user", "content": "Reply with exactly one word: pong"}]
first = complete(messages, refresh=True)  # refresh forces a real API call every run
after_first = usage_summary()
second = complete(messages)
after_second = usage_summary()

assert after_first["api_calls"] == 1, f"first call made {after_first['api_calls']} API calls"
assert after_second["cache_hits"] == after_first["cache_hits"] + 1, "second call was not a cache hit"
assert after_second["api_calls"] == after_first["api_calls"], "second call reached the API"
assert second == first, "cached text differs from the original response"

print("model:", model)
print("response:", repr(second))
print("usage_summary():", usage_summary())
