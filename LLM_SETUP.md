# LLM Integration Setup Guide

## Overview
The simple agent loop now supports using an LLM (via Oumi API) to generate experiment strategies instead of heuristic-based generation.

## Setup Instructions

### 1. Install Dependencies
```bash
cd prometheus-lab
pip install -r requirements.txt
```

### 2. Set Up Oumi API Key
You need to set the `OUMI_API_KEY` environment variable:

```bash
export OUMI_API_KEY='your-api-key-here'
```

To make this permanent, add the above line to your `~/.bashrc` or `~/.zshrc` file.

### 3. Run the Agent Loop with LLM

To run one cycle with LLM-generated strategies:
```bash
python -m src.simple_agent_loop --use-llm
```

To run multiple cycles:
```bash
python -m src.simple_agent_loop --use-llm --max-steps 3
```

To run without LLM (heuristic mode):
```bash
python -m src.simple_agent_loop
```

## How It Works

1. **Prompt Building**: The system reads the prompt template from `prompts/strategy_prompt.md` and injects current experiment results from the `experiments/` directory.

2. **LLM Call**: The full prompt is sent to the Oumi API, which returns a JSON response with proposed strategies.

3. **Strategy Parsing**: The JSON response is parsed into `StrategyConfig` objects using the `llm_adapter.py` module.

4. **Experiment Execution**: The first strategy from the LLM is executed as an experiment, and results are compared to the current best.

## Files Modified/Created

- **Created**: `src/results_text.py` - Formats experiment results for LLM prompts
- **Modified**: `src/llm_client.py` - Implemented TODO with Oumi API integration
- **Modified**: `src/simple_agent_loop.py` - Added `--use-llm` flag support
- **Modified**: `requirements.txt` - Added `oumi-sdk` dependency

## Troubleshooting

### ImportError: oumi-sdk not found
Run: `pip install oumi-sdk`

### ValueError: OUMI_API_KEY not set
Make sure you've exported the API key: `export OUMI_API_KEY='your-key'`

### LLM returns invalid JSON
The system will catch the error and fall back to heuristic strategy generation automatically.