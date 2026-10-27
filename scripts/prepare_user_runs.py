"""Prepare runnable configs with fresh output paths; does not train models."""
from datetime import datetime
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = root / "configs" / "user_runs"
    destination.mkdir(parents=True, exist_ok=True)
    for family in ("pilot", "mechanism", "targeted_controls"):
        plans = json.loads((root / "configs" / f"{family}.json").read_text())
        prepared = []
        for original in plans:
            args = list(original)
            i = args.index("--output") + 1
            previous = Path(args[i])
            args[i] = str(Path("runs") / "user_runs" / tag / previous.parent.name / previous.name)
            prepared.append(args)
        (destination / f"{family}.json").write_text(json.dumps(prepared, indent=2) + "\n")
    expanded = []
    for original in json.loads((root / "configs" / "pilot.json").read_text()):
        for inner_seed in (2027, 3027, 4027, 5027, 6027):
            for seed in range(3):
                args = list(original)
                dataset = args[args.index("--dataset") + 1]
                mode = args[args.index("--mode") + 1]
                args[args.index("--output") + 1] = str(Path("runs") / "user_runs" / tag / "expanded" / f"{dataset}_{mode}_i{inner_seed}_s{seed}")
                args += ["--fold", "0", "--inner-seed", str(inner_seed), "--seed", str(seed)]
                expanded.append(args)
    (destination / "expanded_validation.json").write_text(json.dumps(expanded, indent=2) + "\n")
    print(f"Prepared 20 clinical, 36 mechanism, 4 targeted and 300 expanded-validation run configurations.\nExpanded validation keeps outer fold 0 fixed and varies only inner partitions.\nOutputs will be saved under runs/user_runs/{tag}/\nNo models have been trained by this script.")


if __name__ == "__main__":
    main()
