"""Reproducible pilot runner. Outer test evaluation must be explicitly selected."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import torch

from .baselines import ReferenceGenerator, CTGANAdapter
from .constraints import ConstraintSet, build_constraints, repair_simple, with_unit_confidence
from .data import SCHEMAS, download_uci, nested_splits, normalize, toy_data
from .diffusion import DiffusionConfig, MixedDiffusion
from .evaluation import evaluate
from .preprocessing import Preprocessor
from .structure import dependency_graph


MODES = ("base", "graph", "hypergraph", "energy", "guidance", "full_no_guidance",
         "constant", "late", "full", "clique", "shuffled", "unit_confidence", "repair",
         "copula", "independent", "bootstrap", "ctgan", "tvae", "dense", "tabddpm")


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def source_hash():
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def config_for(args):
    cfg = DiffusionConfig(steps=args.steps, epochs=args.epochs, width=args.width,
        layers=args.layers, batch_size=args.batch_size, seed=args.seed, device=args.device,
        learning_rate=args.learning_rate, clinical_weight=args.clinical_weight,
        guidance=args.guidance, patience=args.patience, numeric_prediction=args.numeric_prediction,
        gated_structure=args.gated_structure, dense_structure=args.mode == "dense",
        paired_initialization=args.paired_initialization)
    if args.mode in ("base", "graph", "hypergraph", "repair"):
        cfg.clinical_weight, cfg.guidance = 0., 0.
    if args.mode == "energy":
        cfg.guidance = 0.
    if args.mode == "guidance":
        cfg.clinical_weight = 0.
    if args.mode == "full_no_guidance":
        cfg.guidance = 0.
    cfg.graph_enabled = args.mode not in ("base", "hypergraph", "energy", "guidance", "repair")
    cfg.hypergraph_enabled = args.mode not in ("base", "graph", "energy", "guidance", "repair", "clique")
    cfg.unit_confidence = args.mode == "unit_confidence"
    cfg.schedule = args.mode if args.mode in ("constant", "late") else "adaptive"
    if cfg.guidance == 0:
        cfg.schedule = "none"
    return cfg


def run(args):
    if args.threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(args.threads)
    schema = SCHEMAS[args.dataset]
    if args.dataset == "toy":
        frame = toy_data(args.toy_rows, seed=7301)  # mechanism seed differs from training seeds
    else:
        path = Path(args.data_dir) / "processed" / f"{args.dataset}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run the download command first")
        frame = normalize(pd.read_csv(path), schema)
    splits = list(nested_splits(frame, schema, folds=args.folds, seed=args.split_seed, inner_seed=args.inner_seed))
    if not 0 <= args.fold < len(splits):
        raise ValueError("Fold index out of range")
    split = splits[args.fold]
    train, val = frame.iloc[split["train"]], frame.iloc[split["validation"]]
    evaluation_partition = "outer_test" if args.evaluate_test else "inner_validation"
    holdout = frame.iloc[split["test"]] if args.evaluate_test else val
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing experiment: {out}")
    out.mkdir(parents=True, exist_ok=True)
    prep = Preprocessor(schema).fit(train)
    constraints = build_constraints(schema, train)
    cfg = config_for(args)
    if cfg.unit_confidence:
        constraints = with_unit_confidence(constraints)
    evaluation_constraints = constraints
    unknown_rules = set(args.withhold_rule) - {r.name for r in constraints.rules}
    if unknown_rules:
        raise ValueError(f"Unknown withheld rules: {sorted(unknown_rules)}")
    constraints = ConstraintSet([r for r in constraints.rules if r.name not in args.withhold_rule])
    write_json(out / "splits.json", {"split_seed": args.split_seed, "inner_seed": args.inner_seed, "folds": splits})
    write_json(out / "constraints.json", constraints.to_dict())
    write_json(out / "evaluation_constraints.json", evaluation_constraints.to_dict())
    write_json(out / "preprocessing.json", prep.to_dict())
    manifest = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "dataset": schema.name, "schema": asdict(schema), "mode": args.mode, "config": asdict(cfg),
        "fold": args.fold, "split_seed": args.split_seed, "evaluation_partition": evaluation_partition,
        "inner_seed": args.inner_seed, "evaluation_version": 2,
        "outer_test_opened": args.evaluate_test, "train_rows": len(train), "validation_rows": len(val),
        "withheld_rules": args.withhold_rule,
        "test_rows": len(split["test"]), "synthetic_rows": len(train), "source_sha256": source_hash(),
        "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "environment": {name: version(name) for name in ("numpy", "pandas", "torch", "scipy", "scikit-learn")},
        "python": platform.python_version(), "platform": platform.platform(), "threads": args.threads,
        "interpretation": "pilot development only" if not args.evaluate_test else "held-out evaluation; inspect full protocol compliance before publication"}
    write_json(out / "manifest.json", manifest)
    try:
        start = time.perf_counter()
        graph, graph_report = dependency_graph(train, prep, seed=args.split_seed + args.fold, bootstraps=args.bootstraps)
        if args.mode == "clique":
            incidence = np.zeros((len(prep.token_names), len(constraints.hyperedges(prep.token_names))))
            for j, edge in enumerate(constraints.hyperedges(prep.token_names)):
                incidence[edge, j] = 1
            clique = incidence @ incidence.T
            np.fill_diagonal(clique, 0)
            graph += clique / max(float(clique.max()), 1)
        if args.mode == "shuffled":
            # Structural negative control; semantic energy rules are unchanged.
            rng = np.random.default_rng(args.split_seed + args.fold)
            permutation = rng.permutation(len(graph))
            graph = graph[permutation][:, permutation]
            graph_report["permutation"] = permutation.tolist()
        write_json(out / "graph.json", {**graph_report, "used_adjacency": graph.tolist()})
        if args.mode in ("copula", "independent", "bootstrap"):
            generator = ReferenceGenerator(schema, args.mode, args.seed).fit(train)
        elif args.mode in ("ctgan", "tvae"):
            generator = CTGANAdapter(schema, args.mode, args.epochs, args.seed).fit(train)
            manifest["ctgan_version"] = version("ctgan")
        elif args.mode == "tabddpm":
            from .external_baselines import TabDDPMAdapter
            generator = TabDDPMAdapter(prep, cfg).fit(train, val)
            write_json(out / "history.json", generator.history)
            generator.save(out / "checkpoint.pt")
            manifest.update(generator.provenance)
            manifest["selected_epoch"] = generator.best_epoch
            manifest["parameters"] = sum(p.numel() for p in generator.model.parameters())
        else:
            generator = MixedDiffusion(prep, graph, constraints, cfg)
            if args.mode == "shuffled" and generator.model.hyper_blocks is not None:
                # Preserve edge size and degrees up to relabeling, break feature semantics.
                with torch.no_grad():
                    for block in generator.model.hyper_blocks:
                        block.incidence.copy_(block.incidence[permutation].clone())
            generator.fit(train, val)
            write_json(out / "history.json", generator.history)
            generator.save(out / "checkpoint.pt")
            manifest["selected_epoch"] = generator.best_epoch
            manifest["parameters"] = sum(p.numel() for p in generator.model.parameters())
        manifest["fit_seconds_including_structure"] = time.perf_counter() - start
        # Identical class counts and row order across every mode and every seed.
        labels = (train[schema.target].to_numpy() == schema.positive).astype(int)
        start = time.perf_counter()
        synthetic = generator.sample(labels, seed=args.seed + 50000)
        negative = next(v for v in train[schema.target].unique() if v != schema.positive)
        synthetic[schema.target] = np.where(labels == 1, schema.positive, negative)
        if args.mode == "repair":
            synthetic.to_csv(out / "synthetic_before_repair.csv", index=False)
            synthetic, changed = repair_simple(synthetic, constraints)
            manifest["repair_changed_row_fraction"] = changed
        manifest["sample_seconds"] = time.perf_counter() - start
        synthetic.to_csv(out / "synthetic.csv", index=False)
        predictions = {}
        metrics = evaluate(train, holdout, synthetic, schema, evaluation_constraints, seed=args.seed, predictions=predictions)
        pd.DataFrame(predictions).to_csv(out / "utility_predictions.csv", index=False)
        write_json(out / "metrics.json", metrics)
        manifest["status"] = "complete"
        write_json(out / "manifest.json", manifest)
        summary = {"output": str(out.resolve()), "dataset": args.dataset, "mode": args.mode,
                   "partition": evaluation_partition,
                   "association_rmse": metrics["fidelity"]["association_rmse"],
                   "constraint_violation_rate": metrics["constraints"]["synthetic"]["any_rule_violation_rate"],
                   "tstr_forest_auroc": metrics["utility"]["forest"]["TSTR"].get("auroc")}
        print(json.dumps(summary), flush=True)
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(out / "manifest.json", manifest)
        raise


def add_run_args(parser):
    parser.add_argument("--dataset", choices=SCHEMAS, required=True)
    parser.add_argument("--mode", choices=MODES, default="base")
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--inner-seed", type=int, default=None, help="Vary development partitions while preserving the outer test set")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--width", type=int, default=48)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=.001)
    parser.add_argument("--clinical-weight", type=float, default=.1)
    parser.add_argument("--guidance", type=float, default=.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--bootstraps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--toy-rows", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--numeric-prediction", choices=("x0", "v"), default="x0")
    parser.add_argument("--gated-structure", action="store_true")
    parser.add_argument("--paired-initialization", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true", help="Open outer test only after freezing study choices")
    parser.add_argument("--withhold-rule", action="append", default=[], help="Exclude a rule from architecture/loss/guidance, retain it in evaluation")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--datasets", nargs="+", choices=[n for n in SCHEMAS if n != "toy"], default=["heart_failure", "ilpd", "ckd", "diabetes"])
    download.add_argument("--data-dir", default="data")
    single = commands.add_parser("run")
    add_run_args(single)
    batch = commands.add_parser("batch")
    batch.add_argument("--config", required=True, help="JSON list of ordinary run argument lists")
    sample = commands.add_parser("sample")
    sample.add_argument("--checkpoint", required=True)
    sample.add_argument("--positive-count", type=int, required=True)
    sample.add_argument("--negative-count", type=int, required=True)
    sample.add_argument("--seed", type=int, default=0)
    sample.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "download":
        for name in args.datasets:
            print(download_uci(name, args.data_dir), flush=True)
    elif args.command == "batch":
        plans = json.loads(Path(args.config).read_text())
        if not isinstance(plans, list):
            raise ValueError("Batch config must be a list")
        for plan in plans:
            if not isinstance(plan, list) or not all(isinstance(v, str) for v in plan):
                raise ValueError("Each batch entry must be a list of CLI argument strings")
            run(parser.parse_args(["run", *plan]))
    elif args.command == "sample":
        if min(args.positive_count, args.negative_count) < 0 or args.positive_count + args.negative_count == 0:
            raise ValueError("Sample counts must be nonnegative and not both zero")
        if Path(args.output).exists():
            raise FileExistsError("Refusing to overwrite sample output")
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if saved.get("model_type") == "official_tabddpm_adapter":
            from .external_baselines import TabDDPMAdapter
            model = TabDDPMAdapter.load(args.checkpoint)
        else:
            model = MixedDiffusion.load(args.checkpoint)
        torch.set_num_threads(2)
        labels = np.r_[np.ones(args.positive_count, dtype=int), np.zeros(args.negative_count, dtype=int)]
        frame = model.sample(labels, args.seed)
        # Positive/negative condition is explicit, no invented original negative label.
        frame["condition_positive"] = labels
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(str(Path(args.output).resolve()))
    else:
        run(args)


if __name__ == "__main__":
    main()
