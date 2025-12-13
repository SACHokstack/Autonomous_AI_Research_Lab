from typing import Dict, Any, List
from .results_store import load_all_runs

def rank_by_ood_accuracy(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(runs, key=lambda r: r["ood"]["accuracy"], reverse=True)

def summarize_best(runs: List[Dict[str, Any]]) -> str:
    ranked = rank_by_ood_accuracy(runs)
    best = ranked[0]
    cfg = best["config"]
    ood = best["ood"]
    id_acc = best["id"]["accuracy"]
    ood_acc = ood["accuracy"]
    gap = ood_acc - id_acc

    result = (
        f"Best robust strategy so far: {cfg['name']}\n"
        f"  ID accuracy:  {id_acc:.3f}\n"
        f"  OOD accuracy: {ood_acc:.3f}\n"
        f"  ID–OOD gap:   {gap:.3f}\n"
    )
    
    wga = ood.get("worst_group_accuracy")
    if wga is not None:
        result += f"  Worst‑group OOD accuracy: {wga:.3f}\n"
    
    return result

if __name__ == "__main__":
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs")
    print(summarize_best(runs))
