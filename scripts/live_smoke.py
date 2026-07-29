"""Live integration smoke: REAL LLM (generator + judge) + REAL BM25 over ClueWeb22-B.

Run: .venv/bin/python scripts/live_smoke.py
No fakes/fallbacks — any failure raises with the real error.
"""
from autosim.runtime import load_env, build_generator, build_judge, build_retrieval
from autosim.schemas import Persona
from autosim.simulator.policy import CognitivePolicy
from autosim.simulator.driver import SessionDriver
from autosim.simulator.llm_query import LLMQueryGen


def main() -> None:
    env = load_env()
    gen = build_generator(env)
    judge = build_judge(env)
    backend = build_retrieval(env)

    print("generator:", gen.provider, "->", gen.complete("Reply with exactly one word: OK").text.strip()[:60])
    print("judge:    ", judge.provider, "->", judge.complete("Reply with exactly one word: READY").text.strip()[:60])

    hits = backend.search("conversational search evaluation metrics", k=3)
    print("\nBM25 over", backend.index, "->", len(hits), "hits")
    for h in hits:
        print(f"  {h.doc_id}  score={h.score:.3f}  {h.text[:70].replace(chr(10),' ')}...")

    persona = Persona(id="t1", attributes={"expertise": "expert"},
                      info_need="conversational search evaluation metrics")
    policy = CognitivePolicy(gen, prompt=(
        "You are simulating a real expert user in a conversational web-search session. "
        "Pick the single next action that such a user would take now."))
    driver = SessionDriver(policy=policy, backend=backend, llm=gen, max_turns=6,
                           query_gen=LLMQueryGen(gen))
    trace = driver.run(persona, session_id="live-1")

    print("\nlive session trace (real LLM-driven):")
    for t in trace.turns:
        utt = t.action.content[:64].replace(chr(10), " ")
        print(f"  turn {t.t}: {t.action.component:12s} {t.action.kind:9s} "
              f"'{utt}'  retrieved={len(t.retrieved_ids)} clicked={t.clicked_ids}")
    print("stopped:", trace.stopped, "expressed_satisfaction:", round(trace.expressed_satisfaction or 0.0, 3))


if __name__ == "__main__":
    main()
