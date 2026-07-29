from pathlib import Path
from autosim.config import RunConfig
from autosim.metrics.distributions import bootstrap_ci
from autosim.experiments.run_experiment import run_experiment

def _arm(curve: list[float], seed: int) -> dict:
    lo, hi = bootstrap_ci(curve, n=300, seed=seed)
    return {"curve": curve, "final_ci": (lo, hi)}

def run_ablation(cfg: RunConfig, *, fixture: Path) -> dict[str, dict]:
    """Claim-A ablation: full loop vs -calibration (drift only) vs static (loop off).
    Each arm reports its curve and a bootstrap CI (spec §10 criterion #4)."""
    full = run_experiment(cfg, fixture=fixture, calibrate=True).csg_curve
    drift = run_experiment(cfg, fixture=fixture, calibrate=False).csg_curve   # no calibration => no gain
    static = [full[0]] * (cfg.k_cycles + 1)                                   # loop disabled
    return {"full": _arm(full, cfg.seed),
            "no_calibration": _arm(drift, cfg.seed),
            "static": _arm(static, cfg.seed)}
