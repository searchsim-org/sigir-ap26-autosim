from autosim.llm.client import LLMResponse


class FakeLLMClient:
    provider = "fake"

    def __init__(self, script: dict[str, str] | None = None, default: str = "", seed: int = 0):
        self.script = script or {}
        self.default = default
        self.seed = seed

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> LLMResponse:
        key = prompt.strip().splitlines()[0] if prompt.strip() else ""
        return LLMResponse(text=self.script.get(key, self.script.get(prompt.strip(), self.default)))
