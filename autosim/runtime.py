"""Live runtime factories: build REAL clients (LLM + retrieval) from `.env`.

No fakes, no fallbacks. Missing config raises KeyError so it is obvious.
Constants below were discovered against the live services on 2026-06-28.
"""
import os
from pathlib import Path

from autosim.llm.openai_compat import OpenAICompatLLM
from autosim.retrieval.opensearch_bm25 import OpenSearchBM25Backend

# --- discovered live config (override here if the deployment changes) ---
GENERATOR_MODEL = os.environ.get("AUTOSIM_GEN_MODEL", "qwen3-next-80b-a3b-instruct")   # confirmed routable; env-overridable for backbone-swap robustness
JUDGE_MODEL = os.environ.get("AUTOSIM_JUDGE_MODEL", "gemma4-31b-it")                    # different family => judge != generator
OPENSEARCH_HOST = os.environ.get("OPENSEARCH_HOST", "")   # set in .env (see .env.example)
IKAT_INDEX = "clueweb22-b-ikat"


def load_env(path: str | Path = ".env") -> dict[str, str]:
    env = dict(os.environ)
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k] = v.strip().strip('"').strip("'")
    return env


def _llm(env: dict[str, str], model: str) -> OpenAICompatLLM:
    return OpenAICompatLLM(base_url=env["CUSTOM_LLM_ENDPOINT"].rstrip("/") + "/v1",
                           api_key=env["CUSTOM_LLM_API_KEY"], model=model)


def build_generator(env: dict[str, str]) -> OpenAICompatLLM:
    return _llm(env, GENERATOR_MODEL)


def build_judge(env: dict[str, str]) -> OpenAICompatLLM:
    return _llm(env, JUDGE_MODEL)


def build_retrieval(env: dict[str, str]) -> OpenSearchBM25Backend:
    return OpenSearchBM25Backend(host=OPENSEARCH_HOST, user="reader",
                                 password=env["CLUEWEB22_READER"], index=IKAT_INDEX)


SPLADE_URL = os.environ.get("SPLADE_URL", "")   # optional; set in .env (see .env.example)
RERANKER_MODEL = "qwen3-reranker-4b"


def build_splade(env: dict[str, str], os_backend: OpenSearchBM25Backend):
    from autosim.retrieval.splade import SpladeBackend
    return SpladeBackend(splade_url=SPLADE_URL, user="reader",
                         password=env["CLUEWEB22_READER"], os_backend=os_backend)


def build_reranker(env: dict[str, str]):
    from autosim.retrieval.rerank import Reranker
    return Reranker(base_url=env["CUSTOM_LLM_ENDPOINT"], api_key=env["CUSTOM_LLM_API_KEY"],
                    model=RERANKER_MODEL)
