from pydantic import BaseModel, Field


class Persona(BaseModel):
    id: str
    attributes: dict[str, str] = {}     # e.g. expertise, verbosity, politeness
    info_need: str
    ptkb: list[str] = []                # personal text knowledge base (iKAT)


class SimState(BaseModel):
    persona: Persona
    history: list[str] = []             # rendered prior turns
    cognitive: dict[str, float] = {}    # satisfaction, frustration, patience in [0,1]


class Action(BaseModel):
    component: str                      # which component produced it (for π_O analysis)
    kind: str                           # query | click | answer_clarif | feedback | satisfaction | stop
    content: str = ""
    meta: dict = {}


class Turn(BaseModel):
    t: int
    state: SimState
    action: Action
    sut_response: str = ""
    retrieved_ids: list[int] = []
    clicked_ids: list[int] = []


class Trace(BaseModel):
    session_id: str
    persona: Persona
    turns: list[Turn] = Field(default_factory=list)
    stopped: bool = False
    expressed_satisfaction: float | None = None
    csg: float | None = None
