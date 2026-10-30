"""Thin wrapper around the Gemini API: pacing, retries, JSON output, and call counting.

Every network call in the project goes through `GeminiClient.json_call`, so:
- free-tier rate limits are respected (MIN_SECONDS_BETWEEN_CALLS),
- temporary errors (429 rate limit, 500/503 overload) are retried with backoff,
- permanent errors (400/401/403/404 = bad request, bad key, model not available) fail immediately,
- `self.api_calls` counts every successful request (part of our compute-cost reporting).
"""
import json
import time

from src.config import MIN_SECONDS_BETWEEN_CALLS

PERMANENT_ERRORS = {400, 401, 403, 404}


class GeminiClient:
    def __init__(self, api_key, min_interval=MIN_SECONDS_BETWEEN_CALLS, max_retries=6):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.api_calls = 0
        self._last = 0.0

    def json_call(self, model, prompt, schema, temperature=0.0, max_retries=None):
        """Send `prompt`, force a JSON answer matching `schema`, return the parsed JSON."""
        max_retries = max_retries or self.max_retries
        for attempt in range(max_retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config={
                        "temperature": temperature,
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    },
                )
                self.api_calls += 1
                return json.loads(r.text)
            except json.JSONDecodeError:
                print(f"  invalid JSON (attempt {attempt}), retrying")
            except Exception as e:  # google.genai.errors.APIError has .code
                code = getattr(e, "code", None)
                if code in PERMANENT_ERRORS:
                    raise
                delay = min(60, 5 * 2 ** attempt)
                print(f"  {code or type(e).__name__} (attempt {attempt}), waiting {delay}s: {str(e)[:90]}")
                time.sleep(delay)
        raise RuntimeError(f"{model} failed after {max_retries} attempts")
