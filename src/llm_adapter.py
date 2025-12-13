import json
from typing import List
from .strategies import StrategyConfig

def parse_llm_strategies(json_str: str) -> List[StrategyConfig]:
    """
    Parse LLM JSON into StrategyConfig list.
    Supports two shapes:
    1) Our target schema with 'strategies' key.
    2) A fallback schema with 'hypotheses' and 'config'.
    """
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("LLM returned non-JSON response:")
        print(json_str[:1000])
        raise RuntimeError(f"Failed to parse LLM JSON: {e}")

    configs: List[StrategyConfig] = []

    # Preferred schema
    if "strategies" in data:
        for s in data["strategies"]:
            cfg = StrategyConfig(
                name=s.get("name", "unnamed"),
                class_weight=s.get("class_weight", None),
                l2_C=s.get("l2_C", 1.0),
                sample_frac=s.get("sample_frac", 1.0),
                undersample_majority=s.get("undersample_majority", False),
            )
            configs.append(cfg)
        return configs

    # Fallback: hypotheses schema
    if "hypotheses" in data:
        for h in data["hypotheses"]:
            name = h.get("name", "unnamed")
            reg_strength = "normal"
            use_group_dro = False

            if "regularization" in name:
                reg_strength = "strong"
            if "group_dro" in name:
                use_group_dro = True

            cfg = StrategyConfig(
                name=name,
                class_weight=None,
                l2_C=1.0,
                sample_frac=1.0,
                undersample_majority=False,
                reg_strength=reg_strength,
                use_group_dro=use_group_dro,
            )

            configs.append(cfg)
        return configs



    # If neither schema found
    print("LLM JSON did not contain 'strategies' or 'hypotheses':")
    print(json_str[:1000])
    return []
