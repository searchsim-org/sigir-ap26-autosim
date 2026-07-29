"""LEVER-4: build a self-contained human-validation study. Samples held-out turns and emits a single
standalone index.html (no server, no dependencies) that shows an annotator two next-user messages --
one real, one simulated -- in random order and asks which is the actual human. Half the trials pit
real vs the WEIGHT-level simulator (the test), half real vs the DEFAULT simulator (an attention/control
condition a human should pass). At the end the annotator downloads a JSON of their choices for scoring.
Run: .venv/bin/python scripts/build_human_study.py [n_per_condition]
"""
import json
import random
import sys
from pathlib import Path

N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rng = random.Random(7)

turns = []
for y in ["2023", "2024"]:
    f = Path(f"data/ikat/{y}/ikat_rv_queries.jsonl")
    if f.exists():
        turns += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


def usable(t, key):
    r, s = (t.get("real") or "").strip(), (t.get(key) or "").strip()
    return r and s and r.lower() != s.lower() and len(r) > 3 and len(s) > 3


pairs = []
for cond, key in [("weight", "weight_sim"), ("default", "default_sim")]:
    cand = [t for t in turns if usable(t, key)]
    rng.shuffle(cand)
    for i, t in enumerate(cand[:N]):
        real_is_a = rng.random() < 0.5
        a, b = (t["real"], t[key]) if real_is_a else (t[key], t["real"])
        pairs.append({"id": f"{cond}_{i}", "condition": cond,
                      "history": (t.get("history") or "")[-400:], "a": a, "b": b,
                      "real_side": "A" if real_is_a else "B"})
rng.shuffle(pairs)
print(f"built {len(pairs)} pairs ({sum(p['condition']=='weight' for p in pairs)} weight, "
      f"{sum(p['condition']=='default' for p in pairs)} default)")

HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>User-message study</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:720px;margin:24px auto;padding:0 16px;color:#1a1a1a;line-height:1.5}
 h2{font-weight:600} .card{border:1px solid #ddd;border-radius:10px;padding:16px;margin:12px 0}
 .ctx{color:#666;font-size:.9em;background:#f7f7f9;border-radius:8px;padding:8px 10px;margin-bottom:12px;white-space:pre-wrap}
 .opt{display:block;width:100%;text-align:left;border:1.5px solid #cbd5e1;border-radius:10px;padding:14px 16px;margin:10px 0;background:#fff;cursor:pointer;font-size:1.05em}
 .opt:hover{border-color:#2563eb;background:#f0f6ff} .bar{height:6px;background:#eee;border-radius:4px;overflow:hidden;margin:8px 0}
 .fill{height:100%;background:#2563eb;width:0} button.nav{background:#2563eb;color:#fff;border:0;border-radius:8px;padding:10px 18px;font-size:1em;cursor:pointer}
 .muted{color:#777;font-size:.9em} code{background:#f1f1f4;padding:1px 5px;border-radius:4px}
</style></head><body>
<div id="app"></div>
<script>
const PAIRS = /*PAIRS_JSON*/;
let i=0, responses=[], t0=0, annot="";
const app=document.getElementById('app');
function consent(){
 app.innerHTML=`<h2>Which message was written by a real person?</h2>
 <p>In a web-search conversation, a user types short messages to a search assistant. On each screen you
 will see <b>two</b> candidate next messages for the same conversation. <b>One was written by an actual
 human user; the other by an AI simulator.</b> Your task: pick the one you think a real human wrote.</p>
 <p class="muted">There are ${PAIRS.length} screens, about 5-8 minutes. Answers are anonymous and used only
 to evaluate a research simulator. There are no risks; you may stop anytime by closing the tab.</p>
 <p>Optional participant code (e.g. Prolific ID): <input id="aid" style="padding:6px;border:1px solid #ccc;border-radius:6px"></p>
 <p><label><input type="checkbox" id="c"> I consent to take part.</label></p>
 <button class="nav" onclick="start()">Start</button>`;
}
function start(){ if(!document.getElementById('c').checked){alert('Please tick consent to start.');return;}
 annot=document.getElementById('aid').value.trim(); i=0; responses=[]; render(); }
function choose(side){
 const p=PAIRS[i]; responses.push({id:p.id,condition:p.condition,chosen:side,real_side:p.real_side,
  correct:(side===p.real_side),rt_ms:Math.round(performance.now()-t0)});
 i++; (i<PAIRS.length)?render():done();
}
function render(){ const p=PAIRS[i]; t0=performance.now();
 app.innerHTML=`<div class="muted">Screen ${i+1} of ${PAIRS.length}</div>
 <div class="bar"><div class="fill" style="width:${100*i/PAIRS.length}%"></div></div>
 <h2>Which message was written by a real human?</h2>
 ${p.history?`<div class="ctx"><b>Conversation so far:</b>\\n${esc(p.history)}</div>`:''}
 <button class="opt" onclick="choose('A')"><b>A.</b> ${esc(p.a)}</button>
 <button class="opt" onclick="choose('B')"><b>B.</b> ${esc(p.b)}</button>
 <p class="muted">Pick the message that sounds like an actual person, not an AI.</p>`;
}
function done(){ const w=responses.filter(r=>r.condition==='weight'),d=responses.filter(r=>r.condition==='default');
 const acc=a=>a.length?(a.filter(r=>r.correct).length/a.length):0;
 const blob=new Blob([JSON.stringify({annot,ts:Date.now(),responses},null,1)],{type:'application/json'});
 const url=URL.createObjectURL(blob);
 app.innerHTML=`<h2>Thank you.</h2><p>You completed all ${responses.length} screens.</p>
 <p><a class="nav" style="text-decoration:none" href="${url}" download="human_study_${annot||'anon'}.json">Download my responses</a></p>
 <p class="muted">Please send the downloaded file to the researcher. (For your interest, your accuracy was
 ${(100*acc(w)).toFixed(0)}% on the test condition and ${(100*acc(d)).toFixed(0)}% on the control.)</p>`;
}
function esc(s){return (s||'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
consent();
</script></body></html>"""

outdir = Path("docs/paper-v3/human_study")
outdir.mkdir(parents=True, exist_ok=True)
(outdir / "index.html").write_text(HTML.replace("/*PAIRS_JSON*/", json.dumps(pairs, ensure_ascii=False)))
(outdir / "pairs_key.json").write_text(json.dumps(pairs, ensure_ascii=False, indent=1))
print(f"wrote {outdir}/index.html (standalone) and pairs_key.json (ground truth)")
