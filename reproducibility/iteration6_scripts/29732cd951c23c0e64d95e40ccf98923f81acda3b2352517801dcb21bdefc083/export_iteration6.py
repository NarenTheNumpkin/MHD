"""Export a recipe only if a full-model variance setting passes every frozen gate."""
import json

from iteration6 import ROOT, REPORT, SAMPLER, dump, sha


selection = json.loads((REPORT/"selection.json").read_text())
chosen = selection["selected_development_candidate"]
if chosen is None:
    print("No full variance setting passed every gate; no replacement recipe exported.")
else:
    assert chosen in ("full_m0.5", "full_m1")
    mix = float(chosen.split("_m")[1])
    parent = ROOT/"runs/iteration5/training/full_prior_i2027_t0/checkpoint.pt"
    out = ROOT/"artifacts/iteration6/recipe.json"
    assert not out.exists(), "Refusing to replace an exported recipe."
    dump(out, dict(variant=chosen, variance_mix=mix, guidance=10., schedule="adaptive", sampling_batch_size=256,
        checkpoint=str(parent.relative_to(ROOT)), checkpoint_sha256=sha(parent), sampler_sha256=sha(SAMPLER),
        selection_sha256=sha(REPORT/"selection.json"), new_training_performed=False,
        interpretation="Gate-qualified development recipe referencing existing weights; not independently or clinically validated."))
    print(str(out))
