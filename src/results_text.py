"""
Module for formatting experiment results into text for LLM prompts.
"""
import json
from pathlib import Path
from typing import List, Dict, Any

EXPERIMENTS_DIR = Path(__file__).resolve().parents[1] / "experiments"


def load_experiment_results() -> List[Dict[str, Any]]:
    """Load all experiment result JSON files."""
    results = []
    
    if not EXPERIMENTS_DIR.exists():
        return results
    
    for json_file in EXPERIMENTS_DIR.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except (json.JSONDecodeError, IOError):
            continue
    
    return results


def results_to_text() -> str:
    """
    Convert experiment results to formatted text for insertion into prompt.
    Returns a markdown table with key metrics.
    """
    results = load_experiment_results()
    
    if not results:
        return "No experiment results available yet."
    
    # Build markdown table
    lines = []
    lines.append("| Experiment | ID Accuracy | OOD Accuracy | Robustness Gap |")
    lines.append("|------------|-------------|--------------|----------------|")
    
    for result in results:
        name = result.get("name", "unknown")
        metrics = result.get("metrics", {})
        
        id_acc = metrics.get("id_accuracy", 0.0) * 100
        ood_acc = metrics.get("ood_accuracy", 0.0) * 100
        gap = abs(id_acc - ood_acc)
        
        lines.append(f"| {name} | {id_acc:.1f}% | {ood_acc:.1f}% | {gap:.1f}% |")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the function
    print(results_to_text())