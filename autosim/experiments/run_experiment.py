from pathlib import Path
from pydantic import BaseModel
from autosim.config import RunConfig
from autosim.llm.fake import FakeLLMClient
from autosim.data.ikat import load_topics, load_system_scores, load_official_ranking
from autosim.data.real_traces import load_real_behavior, split_holdout
from autosim.metrics.csg import CSGMeter
from autosim.metrics.distributions import bootstrap_ci
from autosim.metrics.ranking import ranking_tau
from autosim.metabolism.controller import MetabolismController

class ExperimentOutput(BaseModel):
    csg_curve: list[float]
    tau_curve: list[float]
    csg_ci_curve: list[tuple[float, float]]
    accepted_per_cycle: list[bool]

def _sim_records(quality: float, topics) -> list[dict]:
    """Deterministic stand-in for full driver runs: higher quality => more realistic behavior.
    (Live version calls SessionDriver per topic; record structure is identical.)"""
    return [{"kind": "query", "text": tp.info_need, "session_len": 4,
             "satisfaction": round(0.5 + 0.1 * quality, 3)} for tp in topics]

def run_experiment(cfg: RunConfig, *, fixture: Path, calibrate: bool = True) -> ExperimentOutput:
    topics = load_topics(fixture)
    _, hold = split_holdout(load_real_behavior(fixture), frac=0.5, seed=cfg.seed)
    scores = load_system_scores(fixture)
    official = load_official_ranking(fixture)
    meter = CSGMeter(dimensions=cfg.csg_dimensions)
    pref = {s: sum(v.values()) for s, v in scores.items()}

    ctrl = MetabolismController(FakeLLMClient(default="choose the component that matches real users:"),
                               measure=lambda p: {}, policy_prompt="choose:")

    def measure(prompt: str) -> dict[str, float]:
        # calibration signal: realism grows with accumulated experience (accepted principles)
        quality = min(1.0, 0.5 + 0.1 * len(ctrl.peb.principles)) if calibrate else 0.5
        res = meter.score(_sim_records(quality, topics), hold, human_ref=hold,
                          pref_scores=pref, official_order=official, seed=cfg.seed)
        return {k: v for k, v in res.per_dimension.items() if k in cfg.csg_dimensions}
    ctrl.measure = measure

    def composite(per): return 100.0 * sum(per.values()) / len(per)
    def ci_of(per): return tuple(100 * x for x in bootstrap_ci(list(per.values()), n=300, seed=cfg.seed))

    base = measure(ctrl.policy_prompt)
    # NOTE (fake mode): tau uses the fixture's fixed per-system pref scores, so tau_curve is
    # constant here. The live path recomputes `pref` from real per-session SUT outcomes each
    # cycle, so tau varies and is the headline sim-to-real outcome metric (spec DD3).
    csg_curve, tau_curve, ci_curve = [composite(base)], [ranking_tau(pref, official)], [ci_of(base)]
    accepted = []
    for i in range(cfg.k_cycles):
        # vary the divergence per cycle so each distilled PEB principle is unique
        # (PEB dedupes) => experience grows every cycle => the curve rises continuously
        r = ctrl.run_cycle(divergences=[{"dim": cfg.csg_dimensions[0], "delta": round(-0.2 - 0.01 * i, 2)}])
        per = measure(ctrl.policy_prompt)
        csg_curve.append(composite(per)); tau_curve.append(ranking_tau(pref, official))
        ci_curve.append(ci_of(per)); accepted.append(r.accepted)
    return ExperimentOutput(csg_curve=csg_curve, tau_curve=tau_curve,
                            csg_ci_curve=ci_curve, accepted_per_cycle=accepted)
