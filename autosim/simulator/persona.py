from autosim.schemas import Persona, SimState


def init_state(persona: Persona) -> SimState:
    return SimState(persona=persona, history=[],
                    cognitive={"satisfaction": 0.2, "frustration": 0.0, "patience": 1.0})


def update_latent(state: SimState, clicked_relevant: bool) -> None:
    """Minimal latent-state dynamics: satisfaction rises on relevant clicks; patience decays."""
    s = state.cognitive
    s["satisfaction"] = min(1.0, s.get("satisfaction", 0.0) + (0.3 if clicked_relevant else -0.05))
    s["patience"] = max(0.0, s.get("patience", 1.0) - 0.2)
