"""Copy immutable iteration-2 plans to fresh output paths without running them."""
from datetime import datetime
import json
from pathlib import Path


root = Path(__file__).resolve().parents[1]
tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
destination = root / "configs" / "user_runs"
destination.mkdir(parents=True, exist_ok=True)
for family in ("screen", "replication", "external", "inner", "external_diagnostic"):
    plans = json.loads((root / "configs" / f"iteration2_{family}.json").read_text())
    for args in plans:
        i = args.index("--output") + 1
        old = Path(args[i])
        args[i] = str(Path("runs") / "user_runs" / tag / "iteration2" / old.parent.name / old.name)
    (destination / f"iteration2_{family}.json").write_text(json.dumps(plans, indent=2) + "\n")
    print(f"{family}: {len(plans)} planned runs")
print(f"Fresh paths use runs/user_runs/{tag}/iteration2. No models were trained by this script.")
