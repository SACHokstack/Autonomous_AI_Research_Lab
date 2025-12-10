from src.strategies import StrategyConfig
from src.run_experiment import run_experiment

def main():
    configs = [
        StrategyConfig(name="baseline"),
        StrategyConfig(name="class_balanced", class_weight="balanced"),
        StrategyConfig(name="undersample", undersample_majority=True),
        StrategyConfig(name="undersample_balanced",
                       undersample_majority=True, class_weight="balanced"),
    ]

    for cfg in configs:
        run_experiment(cfg)

if __name__ == "__main__":
    main()
