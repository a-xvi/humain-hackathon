import importlib, src.llm
importlib.reload(src.llm)

def _to_json_schema(schema):
    out = {"type": schema["type"].lower()}
    if "enum" in schema:
        out["enum"] = schema["enum"]
    if "items" in schema:
        out["items"] = _to_json_schema(schema["items"])
    return out


class OpenAIClient:
    """Same interface as GeminiClient, backed by OpenAI structured outputs."""

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
            try:
                kwargs = dict(model=model, messages=[{"role": "user", "content": prompt}], response_format=fmt)
                if not model.startswith(("o", "gpt-5")):
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