from dataclasses import dataclass
from typing import Optional
from dataclasses import dataclass
from typing import Optional

@dataclass
class StrategyConfig:
    name: str
    class_weight: Optional[str] = None
    l2_C: float = 1.0
    sample_frac: float = 1.0
    undersample_majority: bool = False  # NEW
               # subsample fraction of training data
