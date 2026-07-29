from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PersonaExperienceBase:
    """EvolveR-style store of NL behavioral principles, retrieved by lexical similarity (P1)."""

    def __init__(self):
        self.principles: list[str] = []

    def add(self, principle: str) -> None:
        if principle and principle not in self.principles:
            self.principles.append(principle)

    def retrieve(self, query: str, *, k: int = 3) -> list[str]:
        if not self.principles:
            return []
        vec = TfidfVectorizer().fit(self.principles + [query])
        sims = cosine_similarity(vec.transform([query]), vec.transform(self.principles))[0]
        order = sims.argsort()[::-1][:k]
        return [self.principles[i] for i in order]
