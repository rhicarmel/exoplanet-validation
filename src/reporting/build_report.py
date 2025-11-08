import argparse, json, os
from pathlib import Path

TEMPLATE = """# Exoplanet Candidate Validation Findings

## Summary
This report summarizes current model results and feature insights. Replace placeholders once you run training on real data.

## Metrics
```json
{metrics}
```

## Notes
- Calibration and interpretability panels to be added.
- Replace sample data with NASA Exoplanet Archive extracts.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    metrics_json = "{}"
    if os.path.exists(args.metrics):
        metrics_json = Path(args.metrics).read_text()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(TEMPLATE.format(metrics=metrics_json))
    print(f"Wrote report to {out_path}")

if __name__ == "__main__":
    main()
