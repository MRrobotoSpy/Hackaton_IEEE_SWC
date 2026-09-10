"""Ollama helper — 'Explain this scenario' panel.

Reads the current scenarios.json summary and answers user questions grounded
in those numbers. Falls back gracefully if the Ollama server is not running.
"""

from __future__ import annotations

import ollama

DEFAULT_MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """You are ONDA's analyst. You explain the results of three
traffic-signal scenarios (A, B, C) to a judge or a curious citizen.

Rules you MUST follow:
- Base every claim on the numbers you were given. Never invent metrics.
- Scenario A is the untouched baseline (today).
- Scenario B is an efficiency-only optimizer. It usually LOWERS total delay
  but pushes some zones above baseline — that is the fairness cost.
- Scenario C is ONDA: max-pressure signal reallocation + diversion under a
  hard constraint that NO zone ends up worse than A. Tokens are paid to the
  drivers who accepted a diversion or a departure shift.
- The one metric that matters most is `zones_worse`. If it is > 0, the
  scenario is failing the fairness test regardless of efficiency gains.
- If asked about an intersection or zone not present in the data, say so.
- Answer in the same language as the question. Be concise.
"""


def is_available(client: ollama.Client | None = None) -> tuple[bool, str]:
    client = client or ollama.Client()
    try:
        client.list()
        return True, ""
    except Exception as exc:
        return False, str(exc)


def list_models(client: ollama.Client | None = None) -> list[str]:
    client = client or ollama.Client()
    try:
        resp = client.list()
        models = resp.get("models", []) if isinstance(resp, dict) else resp.models
        out = []
        for m in models:
            name = m["name"] if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                out.append(name)
        return out
    except Exception:
        return []


REROUTE_SYSTEM_PROMPT = """You explain a specific driving scenario in which
an ambulance and a private car are computed on a real OSM street network.

Rules you MUST follow:
- Use ONLY the numbers you are given. Do NOT invent multipliers or units.
- The token reward formula is exactly:
    tokens = round(extra_time_min × 10 × equity_multiplier)
  The equity_multiplier is 1.0 unless stated otherwise. Do not guess it.
- The car reroute is triggered ONLY if there is at least one space-time
  conflict (same intersection within ±meeting_window seconds).
- If conflicts == 0, no reroute happens and tokens = 0, regardless of how
  many nodes the two routes share.
- Explain physically, not abstractly. Numbers first, adjectives after.
- Answer in the language of the question. Be concise (3-4 sentences).
"""


def stream_answer(
    question: str,
    scenario_summary: str,
    model: str = DEFAULT_MODEL,
    client: ollama.Client | None = None,
    system_prompt: str | None = None,
):
    """Generator that yields token chunks for st.write_stream."""
    client = client or ollama.Client()
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Here is the current snapshot of the scenario:\n"
                f"```\n{scenario_summary}\n```\n\n"
                f"Question: {question}"
            ),
        },
    ]
    for chunk in client.chat(model=model, messages=messages, stream=True):
        piece = chunk.get("message", {}).get("content") if isinstance(chunk, dict) else None
        if piece is None:
            piece = getattr(getattr(chunk, "message", None), "content", "")
        if piece:
            yield piece
