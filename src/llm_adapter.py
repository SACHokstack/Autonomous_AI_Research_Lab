import json
from typing import List

from .strategies import StrategyConfig

def parse_llm_strategies(json_str: str) -> List[StrategyConfig]:
    """
    Take LLM JSON output and convert to a list of StrategyConfig.
    Expects format:
    {
      "strategies": [ { ... }, ... ],
      "rationale": { ... }
    }
    """
    data = json.loads(json_str)
    strategies = data.get("strategies", [])
    configs = []

    for s in strategies:
        cfg = StrategyConfig(
            name=s.get("name", "unnamed"),
            class_weight=s.get("class_weight", None),
            l2_C=s.get("l2_C", 1.0),
            sample_frac=s.get("sample_frac", 1.0),
            undersample_majority=s.get("undersample_majority", False),
        )
        configs.append(cfg)

    return configs
if __name__ == "__main__":
    fake_response = """
    {
      "strategies": [
        {
          "name": "llm_balanced_strong_reg",
          "class_weight": "balanced",
          "l2_C": 0.5,
          "sample_frac": 1.0,
          "undersample_majority": false
        }
      ],
      "rationale": {
        "llm_balanced_strong_reg": "Example rationale."
      }
    }
    """
    cfgs = parse_llm_strategies(fake_response)
    for c in cfgs:
        print(c)
