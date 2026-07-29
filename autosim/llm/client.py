from typing import Protocol
from pydantic import BaseModel


class LLMResponse(BaseModel):
    text: str
    meta: dict = {}


class LLMClient(Protocol):
    provider: str

    def complete(self, prompt: str, *, system: str = "", temperature: float = 0.0) -> LLMResponse: ...
