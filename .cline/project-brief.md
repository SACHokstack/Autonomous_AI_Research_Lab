# Prometheus Lab Brief

- Python project for ML robustness under distribution shift.
- Core loop:
  - StrategyConfig → run_experiment → experiments/run_*.json → results analysis → agent loop.
- LLM is used to propose new StrategyConfigs based on:
  - problem context,
  - current results,
  - robustness techniques reference.

Important:
- src/simple_agent_loop.py: main agent loop.
- src/llm_client.py: all LLM provider-specific code goes here.
- src/llm_adapter.py: parses LLM JSON into StrategyConfig objects.
