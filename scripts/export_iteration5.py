"""Export only a gate-qualified development candidate with its evaluated guidance."""
from dataclasses import asdict
import json

import numpy as np
import pandas as pd

from iteration5 import ROOT, REPORT, Study, dump, sha
from iteration5_model import ObservationInitializedDiffusion
from medhyperdiff.data import SCHEMAS
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule


selection = json.loads((REPORT/"selection.json").read_text())
variant = selection["selected_development_candidate"]
if variant is None:
    print("No full variant met all correction gates; no replacement sampler exported.")
else:
    assert variant in ("full_prior", "full_prior_w4")
    parent = ROOT/"runs/iteration5/training"/f"{variant}_i2027_t0"/"checkpoint.pt"
    out = ROOT/"artifacts/iteration5"/f"ilpd_{variant}_i2027_t0_g10.pt"
    assert not out.exists(), "Refusing to replace an exported checkpoint."
    model = ObservationInitializedDiffusion.load(parent)
    model.config.guidance = 10.
    model.config.schedule = "adaptive"
    model.schedule = guidance_schedule(model.config.steps, "adaptive")
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    # The ordinary CLI loader must generate the exact evaluated pool.
    loaded = MixedDiffusion.load(out)
    frame, split = Study().data("ilpd", 2027)
    schema = SCHEMAS["ilpd"]
    labels = np.tile((frame.iloc[split["train"]][schema.target] == schema.positive).astype(int), 5)
    generated = loaded.sample(labels, seed=70001, batch_size=256)
    reference = ROOT/"runs/iteration5/evaluations"/f"{variant}_i2027_t0_s70001_n5"/"synthetic.csv"
    saved = pd.read_csv(reference, float_precision="round_trip")
    pd.testing.assert_frame_equal(generated, saved[generated.columns], check_exact=True)
    manifest = dict(status="verified", variant=variant, checkpoint=str(out.relative_to(ROOT)),
        checkpoint_sha256=sha(out), parent_checkpoint=str(parent.relative_to(ROOT)), parent_checkpoint_sha256=sha(parent),
        selection_sha256=sha(REPORT/"selection.json"), config=asdict(model.config),
        observation_weight=model.observation_weight, replayed_table=str(reference.relative_to(ROOT)),
        replay_exact=True, sampling_seed=70001, sampling_batch_size=256, new_training_performed=False,
        interpretation="Gate-qualified development candidate; not independently or clinically validated.")
    dump(out.parent/"manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
