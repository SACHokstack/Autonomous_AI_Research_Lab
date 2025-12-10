from typing import Dict

def is_better(new: Dict, best: Dict, tol: float = 0.01, min_imp: float = 0.02) -> bool:
    """
    Decide if new strategy is better than best, focusing on OOD robustness.
    """
    id_new = new["id"]["accuracy"]
    ood_new = new["ood"]["accuracy"]
    id_best = best["id"]["accuracy"]
    ood_best = best["ood"]["accuracy"]

    # 1) Don't tank ID too much
    if id_new < id_best - tol:
        return False

    # 2) Improve OOD enough
    if ood_new <= ood_best + min_imp:
        return False

    # 3) Prefer smaller ID–OOD gap
    gap_new = abs(id_new - ood_new)
    gap_best = abs(id_best - ood_best)

    return gap_new < gap_best
