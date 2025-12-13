# src/llm_client.py

import os
import json
from typing import List, Tuple
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Optional imports with fallbacks
OUMI_AVAILABLE = False
try:
    from oumi.core.configs import ModelParams, RemoteParams, InferenceConfig
    from oumi.core.types.conversation import Conversation, Message, Role
    from oumi.inference import OpenAIInferenceEngine
    OUMI_AVAILABLE = True
except ImportError:
    pass

GROQ_DIRECT_AVAILABLE = False
try:
    from groq import Groq
    GROQ_DIRECT_AVAILABLE = True
except ImportError:
    pass

from .strategies import StrategyConfig
from .results_text import results_to_text

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "strategy_prompt.md"


def _load_prompt() -> str:
    """Load the strategy prompt and inject current results."""
    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Prompt file not found: {PROMPT_PATH}")
    static_prompt = PROMPT_PATH.read_text(encoding="utf-8")
    results_block = results_to_text()
    return static_prompt.replace("{{EXPERIMENT_RESULTS}}", results_block)


def parse_llm_strategies(raw_text: str) -> List[StrategyConfig]:
    """
    Robust parser that handles real-world LLM output:
    - Strips markdown code blocks
    - Works with or without JSON mode
    - Gracefully skips invalid strategies
    """
    text = raw_text.strip()

    # Remove common markdown wrappers
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    if not text:
        raise ValueError("LLM returned empty response after cleaning")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from LLM:\n{text}\n\nError: {e}")

    # Support multiple formats: { "strategies": [...] }, { "hypotheses": [...] }, or direct list
    strategies_data = data.get("strategies") or data.get("hypotheses") or data
    if not isinstance(strategies_data, list):
        raise ValueError(f"Expected 'strategies' list or list root, got {type(strategies_data)}")

    strategies = []
    for i, hypothesis in enumerate(strategies_data):
        try:
            # Extract the config from the hypothesis structure
            if isinstance(hypothesis, dict) and "config" in hypothesis:
                cfg = hypothesis["config"]
                # Map the LLM's config structure to StrategyConfig format
                strategy_name = hypothesis.get("name", f"strategy_{i}")
                strategy_config = {
                    "name": strategy_name,
                    # Extract parameters from the config if they exist
                    "class_weight": cfg.get("params", {}).get("class_weight"),
                    "l2_C": cfg.get("params", {}).get("l2_C", 1.0),
                    "sample_frac": cfg.get("params", {}).get("sample_frac", 1.0),
                    "undersample_majority": cfg.get("params", {}).get("undersample_majority", False),
                    "reg_strength": cfg.get("params", {}).get("reg_strength", "normal")
                }
                # Remove None values to use defaults
                strategy_config = {k: v for k, v in strategy_config.items() if v is not None}
                strategies.append(StrategyConfig(**strategy_config))
            else:
                # Fallback to direct mapping for backward compatibility
                strategies.append(StrategyConfig(**hypothesis))
        except Exception as e:
            print(f"Warning: Skipping invalid strategy #{i}: {hypothesis} | Error: {e}")

    if not strategies:
        raise ValueError("No valid StrategyConfig objects found in LLM response")

    return strategies


def call_llm_and_get_strategies() -> Tuple[List[StrategyConfig], str]:
    """
    Main entry point used by your agent.
    Returns (strategies_list, rationale_text)
    """
    prompt = _load_prompt()
    rationale = "No rationale extracted."

    # === Try Oumi + OpenAI first (if available) ===
    if OUMI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
        try:
            model_params = ModelParams(model_name="gpt-4o-mini")
            remote_params = RemoteParams(
                api_url="https://api.openai.com/v1",
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            engine = OpenAIInferenceEngine(model_params=model_params, remote_params=remote_params)

            conversation = Conversation(messages=[Message(role=Role.USER, content=prompt)])
            output = engine.infer(input=[conversation], inference_config=InferenceConfig())
            raw = output[0].messages[-1].content

            strategies = parse_llm_strategies(raw)
            if isinstance(output[0].messages[-1].content, dict) and "rationale" in output[0].messages[-1].content:
                rationale = output[0].messages[-1].content.get("rationale", rationale)

            return strategies, rationale
        except Exception as e:
            print(f"Warning: Oumi/OpenAI fallback failed: {e}")

    # === Main path: Groq with native JSON mode ===
    if not GROQ_DIRECT_AVAILABLE:
        raise RuntimeError("Please install Groq SDK: pip install groq")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GROQ_API_KEY in .env file")

    client = Groq(api_key=api_key)

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert ML researcher. "
                        "Respond with a single valid JSON object only. "
                        "Never use markdown. Never add explanations outside the JSON."
                    )
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            response_format={"type": "json_object"},  # Enforced JSON
        )

        message = completion.choices[0].message
        if not message.content:
            raise ValueError("Empty response from Groq")

        raw_text = message.content
        print("\nRaw LLM Output (first 600 chars):")
        print(raw_text[:600])
        if len(raw_text) > 600:
            print("...")

        # Try to extract rationale if present
        try:
            full_data = json.loads(raw_text)
            rationale = full_data.get("rationale", rationale)
        except:
            pass  # Fallback: just use strategies

        strategies = parse_llm_strategies(raw_text)
        print(f"Successfully parsed {len(strategies)} strategies")
        return strategies, rationale

    except Exception as e:
        error_str = str(e).lower()
        if "authentication" in error_str or "401" in error_str:
            raise RuntimeError("Invalid GROQ_API_KEY - check https://console.groq.com/keys")
        if "quota" in error_str or "402" in error_str:
            raise RuntimeError("Groq quota exceeded - check https://console.groq.com/usage")
        raise RuntimeError(f"Groq LLM call failed: {e}")