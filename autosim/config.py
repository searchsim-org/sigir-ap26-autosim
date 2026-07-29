from pathlib import Path
import yaml
from pydantic import BaseModel, model_validator


class RunConfig(BaseModel):
    seed: int = 0
    k_cycles: int = 5
    generator_provider: str = "fake"   # fake | anthropic | openai | vllm
    judge_provider: str = "fake"
    temperature: float = 0.0
    testbed: str = "ikat"
    csg_dimensions: list[str] = ["D1", "D5", "D6"]

    @model_validator(mode="after")
    def _judge_differs(self):
        if self.generator_provider != "fake" and self.generator_provider == self.judge_provider:
            raise ValueError("judge_provider must differ from generator_provider (spec §7)")
        return self


def load_config(path: str | Path) -> RunConfig:
    return RunConfig(**yaml.safe_load(Path(path).read_text()))
