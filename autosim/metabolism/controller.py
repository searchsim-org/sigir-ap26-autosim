from typing import Callable

from pydantic import BaseModel

from autosim.llm.client import LLMClient
from autosim.metabolism.experience_base import PersonaExperienceBase
from autosim.metabolism.clustering import cluster_divergences
from autosim.metabolism.reflect import propose_prompt_edit
from autosim.metabolism.regression_gate import BehaviorBank


def _composite(per: dict[str, float]) -> float:
    return 100.0 * sum(per.values()) / len(per)


class CycleResult(BaseModel):
    accepted: bool
    csg_before: float
    csg_after: float
    policy_prompt: str
    cluster: dict | None = None


class MetabolismController:
    """One applied-layer metabolism cycle. `measure(prompt)->per_dimension` is injected so the
    cycle is testable; the experiment runner supplies the real simulate-and-score closure."""

    def __init__(self, llm: LLMClient, measure: Callable[[str], dict[str, float]], policy_prompt: str):
        self.llm, self.measure, self.policy_prompt = llm, measure, policy_prompt
        self.peb = PersonaExperienceBase()
        self.bank = BehaviorBank()

    def run_cycle(self, *, divergences: list[dict]) -> CycleResult:
        before = self.measure(self.policy_prompt)
        self.bank.record(before)
        clusters = cluster_divergences(divergences)
        target = clusters[0] if clusters else {"dim": "-", "summary": "none"}
        candidate_prompt = propose_prompt_edit(self.llm, current_prompt=self.policy_prompt, cluster=target)
        after = self.measure(candidate_prompt)
        accept = self.bank.accepts(after) and _composite(after) >= _composite(before)
        if accept:
            self.policy_prompt = candidate_prompt
            self.bank.record(after)
            self.peb.add(f"To fix {target['summary']}, prefer the revised policy.")
        return CycleResult(accepted=accept, csg_before=_composite(before),
                           csg_after=_composite(after) if accept else _composite(before),
                           policy_prompt=self.policy_prompt, cluster=target)
