"""Explicit research sampling command for the experimental reverse-variance setting."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from iteration3 import dump, sha
from iteration6_sampler import VarianceSampler
from medhyperdiff.diffusion import guidance_schedule


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--variance-mix", type=float, required=True)
parser.add_argument("--guidance", type=float, default=10.)
parser.add_argument("--positive-count", type=int, required=True)
parser.add_argument("--negative-count", type=int, required=True)
parser.add_argument("--seed", type=int, default=70123)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if min(args.positive_count, args.negative_count) < 0 or args.positive_count+args.negative_count == 0:
    parser.error("Provide nonnegative class counts with positive total.")
if not np.isfinite(args.variance_mix) or not 0 <= args.variance_mix <= 1:
    parser.error("Variance mixture must be finite and in [0, 1].")
if not np.isfinite(args.guidance) or args.guidance < 0:
    parser.error("Guidance must be finite and nonnegative.")
manifest_path = args.output.with_suffix(".manifest.json")
if args.output.exists() or manifest_path.exists():
    raise FileExistsError("Refusing to overwrite samples or their manifest.")
torch.set_num_threads(2)
model = VarianceSampler.load(args.checkpoint)
model.variance_mix = args.variance_mix
model.config.guidance = args.guidance
model.config.schedule = "adaptive" if args.guidance else "none"
model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
labels = np.array([1]*args.positive_count+[0]*args.negative_count)
synthetic = model.sample(labels, seed=args.seed, batch_size=256)
synthetic["condition_positive"] = labels
args.output.parent.mkdir(parents=True, exist_ok=True)
synthetic.to_csv(args.output, index=False)
dump(manifest_path, dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint),
    sampler_sha256=sha(Path(__file__).parent/"iteration6_sampler.py"), variance_mix=args.variance_mix,
    config=asdict(model.config), seed=args.seed, sampling_batch_size=256,
    positive_count=args.positive_count, negative_count=args.negative_count, synthetic_sha256=sha(args.output),
    new_training_performed=False, interpretation="Experimental synthetic research data; no clinical or privacy guarantee."))
print(f"Generated {len(synthetic)} rows at {args.output}")
