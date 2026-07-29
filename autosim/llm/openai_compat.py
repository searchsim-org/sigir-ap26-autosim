"""Real OpenAI-compatible chat client (e.g. a self-hosted LLM gateway).

No fakes, no fallbacks: a failed call raises so the caller sees the real error.
Satisfies the `LLMClient` Protocol (see autosim/llm/client.py).
"""
import time

from openai import OpenAI

from autosim.llm.client import LLMResponse


class OpenAICompatLLM:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 120.0,
                 retries: int = 4):
        self.provider = f"openai_compat:{model}"
        self.model = model
        self.retries = retries
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> LLMResponse:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": prompt}]
        last = None
        for i in range(self.retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature)
                return LLMResponse(text=(resp.choices[0].message.content or ""),
                                   meta={"model": self.model})
            except Exception as e:               # transient -> back off and retry the REAL call
                last = e
                time.sleep(2 * (i + 1))
        raise last
