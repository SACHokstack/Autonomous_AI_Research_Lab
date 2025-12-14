from pathlib import Path
from .auto_analyze import analyze_dataset
from .auto_generate import generate_dataset_spec
from .agent_graph import main

def run_auto_lab(csv_path: str):
    """Full autonomous pipeline."""
    print(f"🚀 Autonomous Fairness Agent: {csv_path}")
    
    # 1. Analyze
    analysis = analyze_dataset(csv_path)
    
    # 2. Generate DatasetSpec
    generate_dataset_spec(csv_path, analysis)
    
    # 3. Run agent
    print("\n🤖 Running agent with auto-generated dataset...")
    main(max_steps=3)
    
    print("\n🎉 Autonomous fairness optimization complete!")

if __name__ == "__main__":
    import sys
    csv_path = sys.argv[1]
    run_auto_lab(csv_path)
