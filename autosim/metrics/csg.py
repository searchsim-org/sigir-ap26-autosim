from pydantic import BaseModel
from autosim.metrics.dimensions import d1_comm_style, d5_satisfaction_ece, d6_stopping
from autosim.metrics.discriminator import discriminator_score
from autosim.metrics.ranking import ranking_tau
from autosim.metrics.distributions import bootstrap_ci

_DIMS = {"D1": d1_comm_style, "D5": d5_satisfaction_ece, "D6": d6_stopping}


class CSGResult(BaseModel):
    composite: float
    per_dimension: dict[str, float]
    ceiling: float | None = None
    ci: tuple[float, float] | None = None


class CSGMeter:
    """Composite Sim2Real gap in [0,100]; higher = closer to real. R reported, not optimized."""

    def __init__(self, dimensions: list[str]):
        self.dimensions = dimensions

    def score(self, sim, real, *, human_ref, pref_scores, official_order, seed: int = 0) -> CSGResult:
        per = {d: _DIMS[d](sim, real) for d in self.dimensions if d in _DIMS}
        per["H"] = 1.0 - discriminator_score(sim, human_ref, seed=seed)   # 1 = indistinguishable
        per["R"] = max(0.0, (ranking_tau(pref_scores, official_order) + 1) / 2)  # map tau in [-1,1] -> [0,1]
        behavioral = [v for k, v in per.items() if k != "R"]
        composite = 100.0 * sum(behavioral) / len(behavioral)
        # human ceiling: real-vs-real on the same dimensions (~1.0), CI via per-dim bootstrap
        ceiling = 100.0 * sum(_DIMS[d](real, human_ref) for d in self.dimensions
                               if d in _DIMS) / max(1, len([d for d in self.dimensions if d in _DIMS]))
        ci = tuple(100 * x for x in bootstrap_ci(behavioral, n=500, seed=seed))
        return CSGResult(composite=composite, per_dimension=per, ceiling=ceiling, ci=ci)
