"""Dump iKAT turns as {ptkb, history, resolved, real} records for weight-level training on a GPU node."""
import json
import os
from pathlib import Path

YEAR = os.environ.get("AUTOSIM_YEAR", "2023")
DATA = Path(f"data/ikat/{YEAR}")
topics = json.loads((DATA / f"{YEAR}_test_topics.json").read_text())
out = []
for tp in topics:
    ptkb = list(tp.get("ptkb", {}).values())
    hist = []
    for t in tp.get("turns", []):
        rec = {"tid": f"{tp.get('number')}_{t.get('turn_id')}",
               "ptkb": ptkb[:8], "history": "\n".join(hist[-4:]) or "(start of conversation)",
               "resolved": t.get("resolved_utterance", t.get("utterance", "")),
               "real": t.get("utterance", "")}
        if rec["real"]:
            out.append(rec)
        hist.append(f"User: {t.get('utterance', '')}")
        if t.get("response"):
            hist.append(f"Assistant: {t['response'][:200]}")
(DATA / "ikat_turns.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out))
print(f"wrote {len(out)} turns -> {DATA/'ikat_turns.jsonl'}")
