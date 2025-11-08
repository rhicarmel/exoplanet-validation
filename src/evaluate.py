import argparse, json, yaml, pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    metrics_path = Path(cfg["paths"]["metrics_file"])
    if not metrics_path.exists():
        print("No metrics found. Run `make train` first.")
        return
    metrics = json.loads(metrics_path.read_text())
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
