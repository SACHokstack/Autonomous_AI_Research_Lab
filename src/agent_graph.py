# agent_graph.py
import os
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq

from .results_store import load_all_runs
from .analyze_results import rank_by_ood_accuracy
from .selection import is_better, run_score   # <-- make sure this line exists
from .strategies import StrategyConfig
from .run_experiment import run_experiment
from .llm_client import call_llm_and_get_strategies



class GraphState(TypedDict):
    best_run: Optional[Dict[str, Any]]
    all_runs: List[Dict[str, Any]]
    proposed_configs: List[Dict[str, Any]]
    step: int
    max_steps: int

    # NEW AGENT FIELDS
    strategy_rationale: str
    research_notes: str
    critic_notes: str
    judge_score: float

# ---------- STRATEGY AGENT (Groq) ----------
STRATEGY_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------- RESEARCH AGENT ----------
RESEARCH_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------- CRITIC AGENT ----------
CRITIC_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.05,
    api_key=os.getenv("GROQ_API_KEY")
)

# ---------- JUDGE AGENT ----------
JUDGE_LLM = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key=os.getenv("GROQ_API_KEY")
)

def load_results_node(state: GraphState) -> GraphState:
    runs = load_all_runs()
    ranked = rank_by_ood_accuracy(runs) if runs else []
    best = ranked[0] if ranked else None
    return {**state, "all_runs": runs, "best_run": best}

def strategy_node(state: GraphState) -> GraphState:
    strategies, rationale = call_llm_and_get_strategies()

    return {
        **state,
        "proposed_configs": [s.__dict__ for s in strategies],
        "strategy_rationale": rationale,
    }

def research_node(state: GraphState) -> GraphState:
    best = state.get("best_run")

    if best is None:
        notes_prompt = "No results yet. Suggest generic improvements."
    else:
        notes_prompt = f"""
Analyze this run like a research scientist.

ID accuracy = {best['id']['accuracy']}
OOD accuracy = {best['ood']['accuracy']}

Give suggestions to reduce OOD gap.
        """

    response = RESEARCH_LLM.invoke(notes_prompt).content
    return {**state, "research_notes": response}

def critic_node(state: GraphState) -> GraphState:
    cfgs = state.get("proposed_configs", [])
    prompt = f"""
You are the CRITIC agent.
Evaluate these proposed strategies:

{cfgs}

Provide short critique and risks.
    """

    response = CRITIC_LLM.invoke(prompt).content
    return {**state, "critic_notes": response}




def run_experiments_node(state: GraphState) -> GraphState:
    best_run = state.get("best_run")

    for cfg_dict in state.get("proposed_configs", []):
        cfg = StrategyConfig(**cfg_dict)
        cand_run = run_experiment(cfg)

        if is_better(cand_run, best_run):
            best_run = cand_run

    return {**state, "best_run": best_run}


def evaluate_node(state: GraphState) -> GraphState:
    runs = load_all_runs()
    ranked = rank_by_ood_accuracy(runs) if runs else []
    best = ranked[0] if ranked else None
    return {**state, "all_runs": runs, "best_run": best}

def judge_node(state: GraphState) -> GraphState:
    best = state.get("best_run")

    # -----------------------------
    # 1. Build the prompt
    # -----------------------------
    if best is None:
        score_prompt = (
            "No experimental runs exist yet.\n"
            "Return ONLY the number 0.0\n"
            "Output format: 0.0"
        )
    else:
        id_acc  = best["id"]["accuracy"]
        ood_acc = best["ood"]["accuracy"]
        wga     = best["ood"].get("worst_group_accuracy", ood_acc)
        gap     = abs(ood_acc - id_acc)

        score_prompt = f"""
You are the judge agent. Your job is to score the run from 0 to 1.

Metrics:
- ID accuracy = {id_acc:.4f}
- OOD accuracy = {ood_acc:.4f}
- Worst group accuracy = {wga:.4f}
- Gap (|OOD - ID|) = {gap:.4f}

Scoring rubric:
- High score (~0.8–1.0) → strong OOD, strong worst-group accuracy, and low gap.
- Medium score (~0.4–0.7) → decent OOD but fairness or gap issues.
- Low score (0–0.3) → poor OOD or bad worst-group accuracy.

IMPORTANT:
Return ONLY a single float between 0 and 1.
No explanation, no text, no words. ONLY the number.

Output format example:
0.72
"""

    # -----------------------------
    # 2. Call LLM
    # -----------------------------
    try:
        raw = JUDGE_LLM.invoke(score_prompt).content.strip()
        score = float(raw)
    except Exception:
        score = 0.0

    return {**state, "judge_score": score}


def decide_continue_node(state: GraphState) -> GraphState:
    step = state.get("step", 0) + 1
    return {**state, "step": step}

def should_continue(state: GraphState) -> str:
    step = state.get("step", 0)
    max_steps = state.get("max_steps", 3)
    judge_score = state.get("judge_score", 0.0)

    if judge_score >= 0.8:
        return END

    if step >= max_steps:
        return END

    return "strategy"

def build_agent_graph():
    builder = StateGraph(GraphState)

    builder.add_node("load_results", load_results_node)
    builder.add_node("strategy", strategy_node)
    builder.add_node("research", research_node)
    builder.add_node("critic", critic_node)
    builder.add_node("run_experiments", run_experiments_node)
    builder.add_node("evaluate", evaluate_node)
    builder.add_node("judge", judge_node)
    builder.add_node("decide_continue", decide_continue_node)

    builder.set_entry_point("load_results")

    builder.add_edge("load_results", "strategy")
    builder.add_edge("strategy", "research")
    builder.add_edge("research", "critic")
    builder.add_edge("critic", "run_experiments")
    builder.add_edge("run_experiments", "evaluate")
    builder.add_edge("evaluate", "judge")
    builder.add_edge("judge", "decide_continue")

    builder.add_conditional_edges(
        "decide_continue",
        should_continue,
        {
            "strategy": "strategy",
            END: END,
        },
    )

    return builder.compile()

def main(max_steps: int = 3):
    graph = build_agent_graph()
    initial: GraphState = {
        "best_run": None,
        "all_runs": [],
        "proposed_configs": [],
        "step": 0,
        "max_steps": max_steps,
        "strategy_rationale": "",
        "research_notes": "",
        "critic_notes": "",
        "judge_score": 0.0,
    }

    final = graph.invoke(initial)
    best = final.get("best_run")

    print("\n=== Final Summary ===")
    print("Steps:", final["step"])
    print("Judge score:", final["judge_score"])
    print("Strategy rationale:", final["strategy_rationale"])
    print("Research notes:", final["research_notes"])
    print("Critic notes:", final["critic_notes"])

    if best:
        print("\nBest Strategy:")
        print(best["config"])
        print(best["id"])
        print(best["ood"])

if __name__ == "__main__":
    main()

