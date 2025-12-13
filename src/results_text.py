from typing import List, Dict, Any

from .results_store import load_all_runs

def results_to_text() -> str:
    """
    Convert experiment results to formatted text for insertion into prompt.
    Returns a markdown-style table with key metrics.
    """
    results: List[Dict[str, Any]] = load_all_runs()

    if not results:
        return "No experiment results available yet."

    lines = []
    lines.append("| Strategy | ID Accuracy | OOD Accuracy | Robustness Gap | Worst Group Acc |")
    lines.append("|----------|-------------|--------------|----------------|-----------------|")

    for r in results:
        cfg = r.get("config", {})
        name = cfg.get("name", "unknown")

        id_acc = r.get("id", {}).get("accuracy", 0.0) * 100
        ood_acc = r.get("ood", {}).get("accuracy", 0.0) * 100
        gap = abs(id_acc - ood_acc)

        wga = r["ood"].get("worst_group_accuracy")
        wga_str = f"{wga:.3f}" if wga is not None else "n/a"

        lines.append(f"| {name} | {id_acc:.1f}% | {ood_acc:.1f}% | {gap:.1f}% | {wga_str} |")

    return "\n".join(lines)

if __name__ == "__main__":
    print(results_to_text())
