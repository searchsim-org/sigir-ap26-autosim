# Human validation study — real vs. simulated user messages

**Goal.** Test whether human annotators can distinguish the weight-level simulator's utterances from
real users', complementing the trained classifiers (embedding-LR ≈0.4, XLM-RoBERTa 0.51) and the
frontier-LLM judges (0.50 / 0.48). If humans are also at chance on the weight-level simulator while
clearly above chance on the default one, the indistinguishability claim is as strong as it can be.

## Task
Two-alternative forced choice. Each screen shows the conversation-so-far and two candidate next user
messages — one **real** (held-out iKAT user), one **simulated**, in random order. The annotator picks
the one written by an actual human. 60 trials per annotator:
- **Test condition (30 trials):** real vs. **weight-level** simulator.
- **Control condition (30 trials):** real vs. **default** simulator. A valid annotator should be
  clearly above chance here; near-chance control performance flags an inattentive annotator.

Trials are shuffled; the real message is on side A/B at random; ground truth is in `pairs_key.json`.

## Design & analysis
- Within-subject (each annotator sees both conditions). Target **N = 30–50** annotators.
- Primary test: mean accuracy in the **test** condition vs. chance (0.5), two-sided binomial / one-sample
  t-test across annotators. Pre-registered prediction: test accuracy ≈ 0.5 (indistinguishable);
  control accuracy ≫ 0.5.
- Secondary: per-annotator accuracy, exclusion of annotators with control accuracy < 0.6 (inattentive),
  inter-annotator agreement, response time.

## Participants & ethics
- Recruit fluent English speakers (e.g., Prolific), anonymous, informed consent on the first screen,
  no personal data collected, no foreseeable risk, may stop anytime. Compensate at a fair hourly rate
  (~6–8 min task). This is a minimal-risk annotation study; check local IRB requirements.

## How to run
1. Host `index.html` anywhere static (or have each annotator open it locally — it is fully
   self-contained, no server or dependencies).
2. Each annotator completes the task and clicks **Download my responses** → sends the JSON back
   (or wire the download step to a form/endpoint of your choice).
3. Collect the JSON files into a folder and score:
   `python scripts/score_human_study.py path/to/responses/*.json`
   (compares each response's `correct` field, aggregates by condition, tests vs. chance.)

## Files
- `index.html` — standalone annotation app (60 pairs embedded).
- `pairs_key.json` — the pairs with ground-truth `real_side` (for reference; the app already scores).
- `../../scripts/build_human_study.py` — regenerate with a different sample size / seed.
- `../../scripts/score_human_study.py` — aggregate collected responses.
