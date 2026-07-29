"""AutoSim as a persona-conditioned user/assessor: given the PTKB persona, the
resolved utterance, and a system's top-k passages, the real LLM reports the
user's satisfaction in [0,1]. Aggregated over turns -> a per-system AutoSim score
whose ranking is correlated (Kendall-tau) with the NIST nDCG ranking.
"""
import re
from autosim.llm.client import LLMClient

# The evolvable variable: the free-text CRITERIA the simulated user applies (robust to evolve;
# the prompt structure and the numeric-output format are fixed).
DEFAULT_CRITERIA = (
    "Considering who YOU are, judge how well these passages let you answer your question. "
    "Reward passages that specifically and faithfully address it; discount tangential or "
    "keyword-only matches."
)
_FORMAT = "Reply with ONLY a number from 0.0 (useless) to 1.0 (fully satisfied)."


def _parse_unit(s: str) -> float:
    m = re.search(r"\d*\.?\d+", s)
    if not m:
        return 0.0
    return max(0.0, min(1.0, float(m.group(0))))


class AutoSimAssessor:
    def __init__(self, llm: LLMClient, criteria: str = DEFAULT_CRITERIA):
        self.llm = llm
        self.criteria = criteria

    def score_turn(self, ptkb: list[str], resolved_utterance: str,
                   passages_text: list[str], *, k: int = 5) -> float:
        persona = "\n".join(f"- {s}" for s in ptkb[:10]) or "- (no profile)"
        psgs = "\n".join(f"[{i + 1}] {t[:300]}" for i, t in enumerate(passages_text[:k])) or "(none)"
        ask = (f"You are simulating a specific user in a conversational search session.\n"
               f"About you:\n{persona}\n\n"
               f"You asked: \"{resolved_utterance}\"\n\n"
               f"The system returned these passages:\n{psgs}\n\n"
               f"{self.criteria}\n{_FORMAT}")
        return _parse_unit(self.llm.complete(ask, temperature=0.0).text.strip())

    def score_system(self, run: dict[str, list], topics: dict[str, dict],
                     turn_ids: list[str], *, k: int = 5) -> float:
        """Mean satisfaction over the given turns (only those present in run+topics)."""
        vals = []
        for tid in turn_ids:
            if tid not in run or tid not in topics:
                continue
            passages = [t for _, t, _ in run[tid][:k]]
            vals.append(self.score_turn(topics[tid]["ptkb"], topics[tid]["resolved"], passages, k=k))
        return sum(vals) / len(vals) if vals else 0.0
