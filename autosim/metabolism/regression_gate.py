class BehaviorBank:
    """Zero-tolerance regression gate: an update must not regress any recorded behavior."""

    def __init__(self, tol: float = 1e-9):
        self.best: dict[str, float] = {}
        self.tol = tol

    def record(self, per_dimension: dict[str, float]) -> None:
        for k, v in per_dimension.items():
            self.best[k] = max(self.best.get(k, v), v)

    def accepts(self, candidate: dict[str, float]) -> bool:
        return all(candidate.get(k, -1) >= b - self.tol for k, b in self.best.items())
