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


def _to_json_schema(schema):
    """Gemini-style schema ({"type": "ARRAY", ...}) -> standard JSON Schema (lowercase types)."""
    out = {"type": schema["type"].lower()}
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "items" in schema:
        out["items"] = _to_json_schema(schema["items"])
    return out


class OpenAIClient:
    """Same interface as GeminiClient (json_call), backed by the OpenAI API with strict structured outputs.
    OpenAI requires an OBJECT at the root, so the schema is wrapped as {"result": ...} and unwrapped again."""

    def __init__(self, api_key, min_interval=0.0, max_retries=6):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.api_calls = 0
        self._last = 0.0

    def list_models(self):
        return sorted(m.id for m in self.client.models.list())

    def json_call(self, model, prompt, schema, temperature=0.0, max_retries=None):
        max_retries = max_retries or self.max_retries
        fmt = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
            "type": "object", "properties": {"result": _to_json_schema(schema)},
            "required": ["result"], "additionalProperties": False}}}
        for attempt in range(max_retries):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], response_format=fmt)
                if not model.startswith(("o", "gpt-5")):  # reasoning models don't accept temperature
                    kwargs["temperature"] = temperature
                r = self.client.chat.completions.create(**kwargs)
                self.api_calls += 1
                return json.loads(r.choices[0].message.content)["result"]
            except json.JSONDecodeError:
                print(f"  invalid JSON (attempt {attempt}), retrying")
            except Exception as e:
                code = getattr(e, "status_code", None)
                if code in PERMANENT_ERRORS:
                    raise
                delay = min(60, 5 * 2 ** attempt)
                print(f"  {code or type(e).__name__} (attempt {attempt}), waiting {delay}s: {str(e)[:90]}")
                time.sleep(delay)
        raise RuntimeError(f"{model} failed after {max_retries} attempts")


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


def client_for(model, get_key):
    """Pick the right API client from the model name. get_key(name) returns a secret, e.g. userdata.get."""
    if model.startswith(("gpt", "o1", "o3", "o4")):
        return OpenAIClient(get_key("OPENAI_API_KEY"))
    return GeminiClient(get_key("GEMINI_API_KEY"))
