"""JSON clients for Groq and Gemini (free tiers) with a jsonl cache, so a run that hits the
daily cap can just be started again the next day."""

import hashlib
import json
import os
import time

import httpx

from pathscope import config

CACHE_DIR = config.DATA_DIR / "llm_cache"

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


class QuotaExhausted(RuntimeError):
    pass


class Cache:
    def __init__(self, name):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.data = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.data[rec["key"]] = rec["value"]

    @staticmethod
    def key(*parts):
        return hashlib.sha1("\x1f".join(parts).encode()).hexdigest()

    def get(self, key):
        return self.data.get(key)

    def put(self, key, value):
        self.data[key] = value
        with self.path.open("a") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")


def _post(url, headers, body, min_gap, tries=6):
    wait = 10.0
    for _ in range(tries):
        time.sleep(min_gap)
        r = httpx.post(url, headers=headers, json=body, timeout=120)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429 or r.status_code >= 500:
            msg = r.text[:300]
            retry = float(r.headers.get("retry-after", 0) or 0)
            # per-day limits come back as 429 with a long retry-after (or say so in the text)
            if retry > 600 or "per day" in msg.lower():
                raise QuotaExhausted(msg)
            time.sleep(max(retry, wait))
            wait = min(wait * 2, 120)
            continue
        raise RuntimeError(f"{r.status_code}: {r.text[:500]}")
    raise QuotaExhausted(f"gave up after {tries} tries at {url}")


def groq_json(system, user, model=GROQ_MODEL, min_gap=2.5):
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (.env)")
    body = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
    }
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
    out = _post(GROQ_URL, {"Authorization": f"Bearer {key}"}, body, min_gap)
    return json.loads(out["choices"][0]["message"]["content"])


def gemini_json(prompt, model=GEMINI_MODEL, min_gap=4.5):
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (.env)")
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }
    out = _post(GEMINI_URL.format(model=model), {"x-goog-api-key": key}, body, min_gap)
    return json.loads(out["candidates"][0]["content"]["parts"][0]["text"])
