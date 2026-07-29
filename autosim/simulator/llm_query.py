"""Real LLM-driven query generation / reformulation.

Drop-in replacement for the deterministic `QueryGen`: uses the real generator
LLM to produce the next search query from the information need, prior queries,
and the current top result. No fakes/fallbacks beyond reusing the info-need if
the model returns an empty string.
"""
from autosim.schemas import SimState, Action
from autosim.llm.client import LLMClient
from autosim.retrieval.backend import Passage


class LLMQueryGen:
    def __init__(self, llm: LLMClient, *, temperature: float = 0.3):
        self.llm = llm
        self.temperature = temperature

    def __call__(self, state: SimState, retrieved: list[Passage]) -> Action:
        prior = [h[len("query:"):].strip() for h in state.history if h.startswith("query:")]
        seen = "; ".join(prior[-4:]) or "(none yet)"
        snippet = retrieved[0].text[:160] if retrieved else "(no results yet)"
        ask = (
            "You are simulating a real user in a web-search session.\n"
            f"Information need: {state.persona.info_need}\n"
            f"Queries already tried: {seen}\n"
            f"Top current result snippet: {snippet}\n"
            "Write the SINGLE next search query (3-8 words) this user would type to make "
            "progress. Do NOT repeat an earlier query. Output ONLY the query text."
        )
        text = self.llm.complete(ask, temperature=self.temperature).text.strip()
        query = text.strip('"').splitlines()[0][:120] if text else state.persona.info_need
        return Action(component="QueryGen", kind="query", content=query)
