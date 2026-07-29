"""Real TREC iKAT loaders: topics (persona/PTKB + resolved utterances), NIST
passage qrels, and iKAT run JSON (turns -> responses -> passage_provenance).
"""
import json
from pathlib import Path


def load_ikat_topics(path: str | Path) -> dict[str, dict]:
    """-> {turn_id 'NUM_t': {ptkb:[str], title, utterance, resolved, gold_response}}"""
    data = json.loads(Path(path).read_text())
    out: dict[str, dict] = {}
    for topic in data:
        num = topic["number"]
        ptkb_raw = topic.get("ptkb", {})
        ptkb = list(ptkb_raw.values()) if isinstance(ptkb_raw, dict) else list(ptkb_raw)
        for t in topic.get("turns") or topic.get("responses", []):   # 2025 uses 'responses'
            tid = f"{num}_{t['turn_id']}"
            utt = t.get("utterance", t.get("user_utterance", ""))    # 2025 uses 'user_utterance'
            out[tid] = {
                "ptkb": ptkb, "title": topic.get("title", ""),
                "utterance": utt,
                "resolved": t.get("resolved_utterance", utt),
                "gold_response": t.get("response", ""),
            }
    return out


def load_qrels(path: str | Path) -> dict[str, dict[str, int]]:
    """TREC qrels 'qid 0 docid rel' -> {turn_id: {passage_id: rel}}"""
    out: dict[str, dict[str, int]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        qid, _, docid, rel = line.split()
        out.setdefault(qid, {})[docid] = int(rel)
    return out


def load_ikat_run(path: str | Path):
    """iKAT run JSON -> (run_name, {turn_id: [(passage_id, text, score)] ranked desc})."""
    d = json.loads(Path(path).read_text())
    name = d.get("run_name") or Path(path).stem
    out: dict[str, list] = {}
    for turn in d.get("turns", []):
        tid = turn["turn_id"]
        seen: dict[str, tuple] = {}
        for resp in turn.get("responses", []):
            for p in resp.get("passage_provenance", []):
                pid = p["id"]
                if pid not in seen:
                    seen[pid] = (p.get("text", ""), float(p.get("score", 0.0)))
        out[tid] = sorted(((pid, t, s) for pid, (t, s) in seen.items()), key=lambda x: -x[2])
    return name, out


def load_runs_dir(dirpath: str | Path) -> dict[str, dict]:
    """Load every *.json run in a directory -> {run_name: {turn_id: ranked passages}}."""
    runs: dict[str, dict] = {}
    for f in sorted(Path(dirpath).glob("*.json")):
        name, r = load_ikat_run(f)
        runs[name] = r
    return runs
