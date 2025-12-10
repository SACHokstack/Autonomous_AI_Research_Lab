from typing import Dict, Any, List
from .results_store import load_all_runs

def rank_by_ood_accuracy(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(runs, key=lambda r: r["ood"]["accuracy"], reverse=True)

def summarize_best(runs: List[Dict[str, Any]]) -> str:
    ranked = rank_by_ood_accuracy(runs)
    best = ranked[0]
    cfg = best["config"]
    id_acc = best["id"]["accuracy"]
    ood_acc = best["ood"]["accuracy"]
    gap = id_acc - ood_acc

    return (
        f"Best robust strategy so far: {cfg['name']}\n"
        f"  ID accuracy:  {id_acc:.3f}\n"
        f"  OOD accuracy: {ood_acc:.3f}\n"
        f"  ID–OOD gap:   {gap:.3f}\n"
    )

if __name__ == "__main__":
    runs = load_all_runs()
    print(f"Loaded {len(runs)} runs")
    print(summarize_best(runs))
