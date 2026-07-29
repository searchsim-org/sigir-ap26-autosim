from autosim.schemas import SimState, Action
from autosim.retrieval.backend import Passage


class QueryGen:
    """Keyword query from the information need (deterministic baseline)."""
    def __call__(self, state: SimState, retrieved: list[Passage]) -> Action:
        need = state.persona.info_need
        return Action(component="QueryGen", kind="query", content=need)


class ClickModel:
    """Position/relevance click: click the top passage above a score threshold."""
    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def __call__(self, state: SimState, retrieved: list[Passage]) -> Action:
        if retrieved and retrieved[0].score >= self.threshold:
            top = retrieved[0]
            return Action(component="ClickModel", kind="click", content=top.text[:80],
                          meta={"did": top.did})
        return Action(component="ClickModel", kind="click", content="", meta={"did": -1})


class StopRule:
    """Stop when satisfaction exceeds a threshold or patience is exhausted."""
    def __init__(self, sat_threshold: float = 0.8):
        self.sat_threshold = sat_threshold

    def __call__(self, state: SimState, retrieved: list[Passage]) -> Action:
        sat = state.cognitive.get("satisfaction", 0.0)
        pat = state.cognitive.get("patience", 1.0)
        stop = sat >= self.sat_threshold or pat <= 0.0
        return Action(component="StopRule", kind="stop" if stop else "continue",
                      content="", meta={"satisfaction": sat, "patience": pat})
