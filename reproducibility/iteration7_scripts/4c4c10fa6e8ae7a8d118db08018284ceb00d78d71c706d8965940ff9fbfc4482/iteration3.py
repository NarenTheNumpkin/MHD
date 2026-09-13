"""Bounded strength sweep with audited checkpoint reuse; no outer-test metrics."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch

from medhyperdiff.cli import source_hash
from medhyperdiff.constraints import repair_simple
from medhyperdiff.data import SCHEMAS, nested_splits, normalize, toy_data
from medhyperdiff.diffusion import MixedDiffusion, guidance_schedule
from medhyperdiff.evaluation import evaluate

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "iteration3"
REPORT = ROOT / "reports" / "iteration3"


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spec(mode, prediction, weight, guidance, repair=False):
    return dict(mode=mode, prediction=prediction, clinical_weight=float(weight),
                guidance=float(guidance), repair=repair)


def name(s):
    return f"{s['mode']}_{s['prediction']}_l{s['clinical_weight']:g}_g{s['guidance']:g}".replace(".", "p") + ("_repair" if s["repair"] else "")


def screen_specs():
    result = [spec("full", p, w, g) for p in ("x0", "v") for w in (0, .01, .1, 1) for g in (0, .2, 2, 10)]
    result += [spec("base", p, 0, g) for p in ("x0", "v") for g in (0, .2, 2, 10)]
    result += [spec("base", p, 0, 0, True) for p in ("x0", "v")]
    return result


class Study:
    def __init__(self):
        torch.set_num_threads(2)
        self.source = source_hash()
        self.index, self.models, self.frames, self.splits = [], {}, {}, {}
        for root in (ROOT / "runs" / "iteration2", RUNS / "training"):
            for path in sorted(root.rglob("manifest.json")):
                m = json.loads(path.read_text())
                if m.get("status") == "complete" and (path.parent / "checkpoint.pt").exists():
                    self.index.append((path.parent, m))

    def data(self, dataset, inner):
        if dataset not in self.frames:
            self.frames[dataset] = (toy_data(512, 7301) if dataset == "toy" else
                normalize(pd.read_csv(ROOT / "data" / "processed" / f"{dataset}.csv"), SCHEMAS[dataset]))
        if (dataset, inner) not in self.splits:
            frame = self.frames[dataset]
            split = next(nested_splits(frame, SCHEMAS[dataset], seed=2026, inner_seed=inner))
            original = next(nested_splits(frame, SCHEMAS[dataset], seed=2026))
            assert split["test"] == original["test"]
            assert not set(split["test"]) & set(split["train"] + split["validation"])
            self.splits[dataset, inner] = split
        return self.frames[dataset], self.splits[dataset, inner]

    def checkpoint(self, dataset, inner, seed, s):
        key = (dataset, inner, seed, s["mode"], s["prediction"], s["clinical_weight"])
        if key in self.models:
            return self.models[key]
        frame, split = self.data(dataset, inner)
        data_hash = hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest()
        needed = dict(steps=30, epochs=100, batch_size=64, width=32, layers=2, heads=4,
            learning_rate=.001, patience=20, seed=seed, clinical_weight=s["clinical_weight"],
            graph_enabled=s["mode"] != "base", hypergraph_enabled=s["mode"] != "base",
            unit_confidence=False, numeric_prediction=s["prediction"], gated_structure=False,
            dense_structure=s["mode"] == "dense", paired_initialization=True)
        for path, m in self.index:
            if m.get("source_sha256") != self.source or m.get("dataset") != dataset or m.get("mode") != s["mode"]:
                continue
            if m.get("outer_test_opened") or m.get("fold") != 0 or m.get("split_seed") != 2026 or m.get("withheld_rules"):
                continue
            if m.get("data_sha256") != data_hash or any(m["config"].get(k) != v for k, v in needed.items()):
                continue
            saved = json.loads((path / "splits.json").read_text())["folds"][0]
            graph = json.loads((path / "graph.json").read_text())
            if saved != split or graph["bootstraps"] != 10:
                continue
            self.models[key] = (path, m)
            return path, m
        train_name = f"{dataset}_{s['mode']}_{s['prediction']}_l{s['clinical_weight']:g}_i{inner}_s{seed}".replace(".", "p")
        out = RUNS / "training" / train_name
        args = [sys.executable, "-m", "medhyperdiff.cli", "run", "--dataset", dataset,
            "--mode", s["mode"], "--numeric-prediction", s["prediction"], "--clinical-weight", str(s["clinical_weight"]),
            "--guidance", "0", "--paired-initialization", "--epochs", "100", "--steps", "30", "--width", "32",
            "--layers", "2", "--batch-size", "64", "--bootstraps", "10", "--seed", str(seed),
            "--inner-seed", str(inner), "--output", str(out)]
        RUNS.mkdir(parents=True, exist_ok=True)
        with (RUNS / "training.log").open("a") as log:
            subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        m = json.loads((out / "manifest.json").read_text())
        dump(out / "invocation.json", args)
        self.index.append((out, m))
        self.models[key] = out, m
        return out, m

    def sample_evaluate(self, dataset, inner, seed, s):
        out = RUNS / "evaluations" / f"{dataset}_{name(s)}_i{inner}_s{seed}"
        identity = dict(dataset=dataset, inner_seed=inner, seed=seed, setting=s)
        if (out / "manifest.json").exists():
            saved = json.loads((out / "manifest.json").read_text())
            assert saved["status"] == "complete" and all(saved[k] == v for k, v in identity.items())
            assert saved["source_sha256"] == self.source
            assert sha(ROOT / saved["parent_checkpoint"]) == saved["parent_checkpoint_sha256"]
            return str(out.relative_to(ROOT))
        parent, parent_m = self.checkpoint(dataset, inner, seed, s)
        model = MixedDiffusion.load(parent / "checkpoint.pt")
        frame, split = self.data(dataset, inner)
        schema = SCHEMAS[dataset]
        train, validation = frame.iloc[split["train"]], frame.iloc[split["validation"]]
        labels = (train[schema.target].to_numpy() == schema.positive).astype(int)
        model.config.guidance = s["guidance"]
        model.config.schedule = "adaptive" if s["guidance"] else "none"
        model.schedule = guidance_schedule(model.config.steps, model.config.schedule)
        manifest = {**identity, "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": self.source, "runner_sha256": sha(Path(__file__)),
            "parent_checkpoint": str((parent / "checkpoint.pt").relative_to(ROOT)),
            "parent_checkpoint_sha256": sha(parent / "checkpoint.pt"),
            "training_config": parent_m["config"], "sampling_config": asdict(model.config),
            "sampling_seed": seed + 50000, "outer_test_opened": False, "evaluation_partition": "inner_validation",
            "selected_epoch": parent_m["selected_epoch"], "parameters": parent_m["parameters"],
            "data_sha256": parent_m["data_sha256"], "evaluation_version": 2, "new_training_performed_for_this_evaluation": False,
            "interpretation": "sampling/evaluation of a referenced fitted model; not an independent model training run"}
        dump(out / "manifest.json", manifest)
        dump(out / "split.json", split)
        try:
            start = time.perf_counter()
            generated = model.sample(labels, seed=seed + 50000)
            negative = next(v for v in train[schema.target].unique() if v != schema.positive)
            generated[schema.target] = np.where(labels == 1, schema.positive, negative)
            if s["repair"]:
                generated.to_csv(out / "before_repair.csv", index=False)
                generated, fraction = repair_simple(generated, model.constraints)
                manifest["repair_changed_fraction"] = fraction
            manifest["sample_seconds"] = time.perf_counter() - start
            generated.to_csv(out / "synthetic.csv", index=False)
            predictions = {}
            start = time.perf_counter()
            metrics = evaluate(train, validation, generated, schema, model.constraints, seed, predictions)
            manifest["evaluation_seconds"] = time.perf_counter() - start
            pd.DataFrame(predictions).to_csv(out / "utility_predictions.csv", index=False)
            dump(out / "metrics.json", metrics)
            manifest["status"] = "complete"
            dump(out / "manifest.json", manifest)
            print(json.dumps({"run": str(out.relative_to(ROOT)), "rf": metrics['utility']['forest']['TSTR']['auroc'],
                "violation": metrics['constraints']['synthetic']['any_rule_violation_rate']}), flush=True)
            return str(out.relative_to(ROOT))
        except Exception as error:
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            dump(out / "manifest.json", manifest)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("screen", "replicate"))
    args = parser.parse_args()
    study = Study()
    plans = []
    if args.stage == "screen":
        for dataset in ("ilpd", "toy"):
            for setting in screen_specs():
                plans.append(dict(dataset=dataset, inner=2027, seed=0, s=setting))
    else:
        selected = json.loads((REPORT / "selection.json").read_text())
        for dataset, settings in selected["selected"].items():
            comparisons = [spec("full", "x0", .1, .2)]
            for s in settings:
                comparisons += [s, spec("base", s["prediction"], 0, 0)]
                if s["mode"] == "full":
                    comparisons.append({**s, "mode": "dense"})
            unique = {name(s): s for s in comparisons}
            for inner in (2027, 3027, 4027):
                for seed in (0, 1, 2):
                    for setting in unique.values():
                        plans.append(dict(dataset=dataset, inner=inner, seed=seed, s=setting))
    dump(ROOT / "configs" / f"iteration3_{args.stage}.json", plans)
    outputs = [study.sample_evaluate(p["dataset"], p["inner"], p["seed"], p["s"]) for p in plans]
    dump(REPORT / f"{args.stage}_registry.json", outputs)


if __name__ == "__main__":
    main()
