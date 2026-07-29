from autosim.schemas import SimState, Action
from autosim.llm.client import LLMClient

COGNITIVE_KINDS = {
    "AnswerClarification": "answer_clarif",
    "GiveFeedback": "feedback",
    "ExpressSatisfaction": "satisfaction",
}

class CognitiveAgent:
    """LLM-backed conversational behaviors. Prompts are fields so the loop can evolve them."""
    def __init__(self, llm: LLMClient, prompts: dict[str, str] | None = None):
        self.llm = llm
        self.prompts = prompts or {k: f"As the user, {k}:" for k in COGNITIVE_KINDS}

    def act(self, component: str, state: SimState, sut_response: str) -> Action:
        prompt = (f"{self.prompts.get(component, component)}\n"
                  f"need={state.persona.info_need}\nlast_system={sut_response}\n")
        text = self.llm.complete(prompt, temperature=0.0).text
        return Action(component=component, kind=COGNITIVE_KINDS[component], content=text)
