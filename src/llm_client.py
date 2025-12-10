from typing import List
from pathlib import Path
import os
import json

from .results_text import results_to_text
from .llm_adapter import parse_llm_strategies
from .strategies import StrategyConfig

try:
    from oumi.core.clients import OumiClient
    OUMI_AVAILABLE = True
except ImportError:
    OUMI_AVAILABLE = False

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "strategy_prompt.md"

def build_full_prompt() -> str:
    """
    Concatenate static prompt (Sections 1,3,4) + dynamic results (Section 2).
    """
    static_prompt = PROMPT_PATH.read_text()
    results_block = results_to_text()
    return static_prompt.replace("{{EXPERIMENT_RESULTS}}", results_block)

def call_llm_and_get_strategies() -> List[StrategyConfig]:
    """
    Call your chosen LLM (Oumi/Cline/etc.) and return StrategyConfigs.
    """
    prompt = build_full_prompt()

    if not OUMI_AVAILABLE:
        raise ImportError(
            "Oumi SDK not installed. Install it with: pip install oumi-sdk"
        )
    
    # Get API key from environment
    api_key = os.environ.get("OUMI_API_KEY")
    if not api_key:
        raise ValueError(
            "OUMI_API_KEY environment variable not set. "
            "Please set it with: export OUMI_API_KEY='your-api-key'"
        )
    
    # Initialize Oumi client
    client = OumiClient(api_key=api_key)
    
    try:
        # Call the LLM API
        response = client.chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are an AI research assistant helping to design machine learning experiments. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="gpt-4",  # Adjust model as needed
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract the response text
        json_str = response.choices[0].message.content
        
        # Parse and return strategies
        configs = parse_llm_strategies(json_str)
        return configs
        
    except Exception as e:
        raise RuntimeError(f"Error calling Oumi API: {str(e)}")