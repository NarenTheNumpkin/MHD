"""Knowledge-equivalent guided-backbone and original-backbone replication controls."""
import json
from iteration3 import Study, REPORT, ROOT, spec, name, dump

selected = json.loads((REPORT / "selection.json").read_text())["selected"]
study, plans = Study(), []
for dataset, settings in selected.items():
    controls = [spec("base", "x0", 0, 0)]
    controls += [spec("base", s["prediction"], 0, s["guidance"]) for s in settings if s["mode"] == "full"]
    for setting in {name(s): s for s in controls}.values():
        for inner in (2027, 3027, 4027):
            for seed in (0, 1, 2):
                plans.append(dict(dataset=dataset, inner=inner, seed=seed, s=setting))
dump(ROOT / "configs" / "iteration3_controls.json", plans)
outputs = [study.sample_evaluate(p["dataset"], p["inner"], p["seed"], p["s"]) for p in plans]
dump(REPORT / "controls_registry.json", outputs)
