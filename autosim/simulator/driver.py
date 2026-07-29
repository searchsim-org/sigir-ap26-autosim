from autosim.schemas import Persona, Turn, Trace
from autosim.simulator.persona import init_state, update_latent
from autosim.simulator.components import QueryGen, ClickModel, StopRule
from autosim.simulator.cognitive import CognitiveAgent, COGNITIVE_KINDS


class SessionDriver:
    def __init__(self, *, policy, backend, llm, max_turns: int = 12, query_gen=None):
        self.policy, self.backend, self.max_turns = policy, backend, max_turns
        self.det = {"QueryGen": query_gen or QueryGen(), "ClickModel": ClickModel(), "StopRule": StopRule()}
        self.cog = CognitiveAgent(llm)

    def run(self, persona: Persona, *, session_id: str) -> Trace:
        state = init_state(persona)
        turns, retrieved, last_resp, stopped = [], [], "", False
        for t in range(self.max_turns):
            comp = self.policy.select(state, turn=t, has_results=bool(retrieved))
            if comp == "StopRule" and self.det["StopRule"](state, retrieved).kind == "stop":
                stopped = True; break
            if comp in self.det and comp != "StopRule":
                action = self.det[comp](state, retrieved)
            elif comp in COGNITIVE_KINDS:
                action = self.cog.act(comp, state, last_resp)
            else:
                action = self.det["QueryGen"](state, retrieved)
            rids, cids = [], []
            if action.kind == "query":
                hits = self.backend.search(action.content, k=10)
                retrieved = hits; rids = [h.did for h in hits]
                last_resp = hits[0].text if hits else ""
                state.history.append("query: " + action.content)
            elif action.kind == "click":
                did = action.meta.get("did", -1)
                if did >= 0:
                    cids = [did]
                    update_latent(state, clicked_relevant=retrieved and retrieved[0].did == did)
            turns.append(Turn(t=t, state=state.model_copy(deep=True), action=action,
                              sut_response=last_resp, retrieved_ids=rids, clicked_ids=cids))
        return Trace(session_id=session_id, persona=persona, turns=turns,
                     stopped=stopped, expressed_satisfaction=state.cognitive.get("satisfaction"))
