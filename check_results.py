import json, glob, os

exp_dir = "experiments"

def load(name_substr):
    files = glob.glob(os.path.join(exp_dir, "run_*.json"))
    for p in files:
        if name_substr in os.path.basename(p):
            with open(p) as f:
                return os.path.basename(p), json.load(f)
    return None, None

base_name, base = load("baseline")
dro_name, dro = load("group_dro_with_early_stopping")

for name, run in [(base_name, base), (dro_name, dro)]:
    if run is None: 
        continue
    print("\n=== ", name, " ===")
    print("ID acc:", run["id"]["accuracy"])
    print("OOD acc:", run["ood"]["accuracy"])
    try:
        print("Worst-group OOD acc:", run["ood"]["worst_group_accuracy"])
    except KeyError:
        print("Worst-group OOD acc: N/A")
