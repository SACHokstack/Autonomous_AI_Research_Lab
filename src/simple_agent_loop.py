import argparse
from copy import deepcopy

from src.strategies import StrategyConfig
from src.run_experiment import run_experiment
from src.results_store import load_all_runs
from src.analyze_results import rank_by_ood_accuracy
from src.selection import is_better
from src.llm_client import call_llm_and_get_strategies

def propose_next_strategy(best_cfg: StrategyConfig, step: int) -> StrategyConfig:
    """
    Very simple heuristic proposer: tweak l2_C and sample_frac.
    """
    new_cfg = deepcopy(best_cfg)
    new_cfg.name = f"auto_step_{step}"

    if step % 2 == 0:
        new_cfg.l2_C *= 0.5  # stronger regularization
    else:
        new_cfg.sample_frac = min(1.0, best_cfg.sample_frac + 0.1)

    return new_cfg

def main(max_steps: int = 5, use_llm: bool = False):
    runs = load_all_runs()
    if not runs:
        # start from baseline if no runs
        cfg = StrategyConfig(name="baseline")
        best_run = run_experiment(cfg)
        runs = [best_run]
    ranked = rank_by_ood_accuracy(runs)
    best_run = ranked[0]
    best_cfg = StrategyConfig(**best_run["config"])

    print("Starting from best:", best_cfg)

    for step in range(1, max_steps + 1):
        if use_llm:
            print(f"\nStep {step}: Calling LLM to generate strategies...")
            try:
                llm_strategies = call_llm_and_get_strategies()
                if not llm_strategies:
                    print("LLM returned no strategies, falling back to heuristic.")
                    cand_cfg = propose_next_strategy(best_cfg, step)
                else:
                    # Use the first strategy from LLM
                    cand_cfg = llm_strategies[0]
                    print(f"LLM proposed strategy: {cand_cfg}")
            except Exception as e:
                print(f"Error calling LLM: {e}")
                print("Falling back to heuristic strategy generation.")
                cand_cfg = propose_next_strategy(best_cfg, step)
        else:
            cand_cfg = propose_next_strategy(best_cfg, step)
        
        cand_run = run_experiment(cand_cfg)

        if is_better(cand_run, best_run):
            print(f"Step {step}: Found better strategy:", cand_cfg)
            best_run = cand_run
            best_cfg = cand_cfg
        else:
            print(f"Step {step}: Candidate not better, keeping current best.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the simple agent loop for ML experiment optimization"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Use LLM (via Oumi API) to generate strategies instead of heuristics"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Maximum number of optimization steps to run (default: 1 for --use-llm)"
    )
    
    args = parser.parse_args()
    
    # When using LLM, default to 1 step unless specified
    max_steps = args.max_steps if args.max_steps != 1 or not args.use_llm else 1
    
    main(max_steps=max_steps, use_llm=args.use_llm)