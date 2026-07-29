from autosim.schemas import SimState
from autosim.llm.client import LLMClient

COMPONENTS = ["QueryGen", "ClickModel", "AnswerClarification", "GiveFeedback",
              "ExpressSatisfaction", "StopRule"]

class RulePolicy:
    """Deterministic baseline π_O (interpretable)."""
    def select(self, state: SimState, *, turn: int, has_results: bool) -> str:
        if not has_results:
            return "QueryGen"
        if state.cognitive.get("satisfaction", 0.0) >= 0.8:
            return "StopRule"
        return "ClickModel"

class CognitivePolicy:
    """LLM-reasoned π_O. `prompt` is the evolvable decision variable (metabolism target)."""
    def __init__(self, llm: LLMClient, prompt: str):
        self.llm = llm
        self.prompt = prompt

    def select(self, state: SimState, *, turn: int, has_results: bool) -> str:
        sat = state.cognitive.get("satisfaction", 0.0)
        ask = (
            f"{self.prompt}\n"
            f"Turn {turn}. You {'have search results to examine' if has_results else 'have no results yet'}. "
            f"Your satisfaction so far: {sat:.1f}/1.0.\n"
            "Choose your next action:\n"
            "- QueryGen: issue or reformulate a search query\n"
            "- ClickModel: open the most relevant result to read it\n"
            "- StopRule: end the session (satisfied, or giving up)\n"
            "Guidance: with unexamined results and low satisfaction, prefer ClickModel; "
            "when satisfied, choose StopRule. Answer with ONLY the component name on the final line."
        )
        reply = self.llm.complete(ask, temperature=0.0).text.strip()
        last = reply.splitlines()[-1].strip() if reply else ""
        for hay in (last, reply):                 # prefer the model's final-line decision
            if hay in COMPONENTS:
                return hay
            for c in COMPONENTS:
                if c.lower() in hay.lower():
                    return c
        return "QueryGen" if not has_results else "ClickModel"   # safe fallback
