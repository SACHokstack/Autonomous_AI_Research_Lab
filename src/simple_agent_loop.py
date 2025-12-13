import argparse

from src.results_store import load_all_runs
from src.analyze_results import rank_by_ood_accuracy
from src.selection import is_better
from src.strategies import StrategyConfig
from src.run_experiment import run_experiment
from src.llm_client import call_llm_and_get_strategies

def main(use_llm: bool = False, max_steps: int = 5):
    runs = load_all_runs()
    if not runs:
        # ensure at least one baseline run exists
        base = run_experiment(StrategyConfig(name="baseline"))
        runs = [base]

    ranked = rank_by_ood_accuracy(runs)
    best_run = ranked[0]
    best_cfg = StrategyConfig(**best_run["config"])
    print("Starting from best:", best_cfg)

    if use_llm:
        # LLM-driven proposals with fallback
        try:
            cfgs = call_llm_and_get_strategies()
        except (ImportError, ValueError, RuntimeError) as e:
            print(f"LLM not available or failed ({e}). Falling back to no new strategies.")
            cfgs = []

        for cfg in cfgs:
            cand_run = run_experiment(cfg)
            if is_better(cand_run, best_run):
                print("Found better strategy via LLM:", cfg)
                best_run = cand_run
                best_cfg = cfg
    else:
        # (optional) keep your heuristic loop here, or just no-op
        print("LLM disabled; no new strategies proposed in this mode.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5)
    args = parser.parse_args()
    main(use_llm=args.use_llm, max_steps=args.max_steps)
