from autosim.llm.client import LLMClient


def propose_prompt_edit(llm: LLMClient, *, current_prompt: str, cluster: dict) -> str:
    """Reflective prompt evolution: ask the LLM to revise the policy prompt to fix one cluster."""
    ask = (f"reflect-edit\nCurrent policy prompt:\n{current_prompt}\n"
           f"Observed failure: {cluster.get('summary', cluster)}\n"
           f"Return ONLY the revised policy prompt that reduces this failure.")
    text = llm.complete(ask, temperature=0.0).text.strip()
    return text or current_prompt
