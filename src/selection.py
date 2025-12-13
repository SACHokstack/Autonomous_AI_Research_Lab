from typing import Dict, Optional, Any

# Hard baselines from run_baseline_test.json
BASELINE_WGA = 0.526
BASELINE_OOD = 0.586

def run_score(run: Dict[str, Any]) -> float:
    """
    Scoring function for agent selection.
    
    Rubric:
    - Must not regress significantly on OOD accuracy or Worst-Group Accuracy (WGA) compared to baseline.
    - If valid, score is weighted combination of WGA (primary) and OOD (secondary).
    """
    id_acc = run["id"]["accuracy"]
    ood_acc = run["ood"]["accuracy"]
    wga = run["ood"].get("worst_group_accuracy", ood_acc)
    
    # Gap is less important if WGA is high, but still good to track
    gap = abs(id_acc - ood_acc)

    # 1. Hard constraints (with small tolerance)
    # If a run drops WGA or OOD below baseline, it's not a "better" strategy for our goal
    if wga < BASELINE_WGA - 0.005 or ood_acc < BASELINE_OOD - 0.005:
        return -1.0

    # 2. Nonlinear scoring to reward high WGA
    # We really want to boost the worst group. 
    # Squaring WGA makes improvements in the 0.5->0.6 range worth more than 0.2->0.3.
    
    alpha = 0.8   # Emphasis on Worst Group
    beta = 0.18   # Emphasis on overall OOD
    gamma = 0.02  # Penalty for ID-OOD gap

    score = alpha * (wga ** 2) + beta * ood_acc - gamma * gap
    return score

def is_better(new_run: Dict[str, Any], best_run: Optional[Dict[str, Any]], min_imp: float = 0.001) -> bool:
    """
    Returns True if new_run has a higher score than best_run.
    """
    if best_run is None:
        # Only accept if it meets the baseline floor
        return run_score(new_run) > 0.0
    
    return run_score(new_run) > run_score(best_run) + min_imp
