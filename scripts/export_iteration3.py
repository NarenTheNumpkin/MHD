"""Package frozen screen-selected samplers; exporting does not train or certify them."""
import json
from pathlib import Path

from iteration3 import ROOT, RUNS, REPORT, name, dump, sha
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule


selected = json.loads((REPORT / "selection.json").read_text())["selected"]
out = ROOT / "artifacts" / "iteration3"
out.mkdir(parents=True, exist_ok=True)
exports = []
for dataset, settings in selected.items():
    for setting in settings:
        evaluation = RUNS / "evaluations" / f"{dataset}_{name(setting)}_i2027_s0"
        m = json.loads((evaluation / "manifest.json").read_text())
        parent = ROOT / m["parent_checkpoint"]
        assert sha(parent) == m["parent_checkpoint_sha256"]
        model = MixedDiffusion.load(parent)
        model.config.guidance = setting["guidance"]
        model.config.schedule = "adaptive" if setting["guidance"] else "none"
        model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
        target = out / f"{dataset}_{name(setting)}_i2027_s0.pt"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite export {target}")
        model.save(target)
        exports.append(dict(checkpoint=str(target.relative_to(ROOT)), sha256=sha(target),
            parent_checkpoint=m["parent_checkpoint"], parent_sha256=m["parent_checkpoint_sha256"],
            setting=setting, dataset=dataset, seed=0, inner_seed=2027,
            status="development sampler selected on the screening split; not an independent-test winner"))
dump(out / "manifest.json", exports)
print(json.dumps(exports, indent=2))
