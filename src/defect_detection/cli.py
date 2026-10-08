"""Command-line entry point: ``vdd <command>`` (or ``python -m defect_detection``).

Heavy imports (torch, pandas, matplotlib) happen inside commands so that the serving
image - which has none of them - can still use ``vdd predict`` / ``vdd serve``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

from defect_detection.logging_utils import configure_logging

log = logging.getLogger("defect_detection.cli")


def _cfg(args):
    import yaml

    from defect_detection.config import load_config

    overrides = {}
    for item in args.set or []:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    return load_config(args.config, overrides)


def cmd_download(args) -> None:
    from defect_detection.data.download import download_mvtec

    print(json.dumps(download_mvtec(args.category, Path(args.data_dir)), indent=2))


def cmd_prepare(args) -> None:
    from defect_detection.data.analyze import run_eda
    from defect_detection.data.manifest import build_manifest
    from defect_detection.data.split import make_split, split_summary

    cfg = _cfg(args)
    raw = cfg.data.raw_dir
    masks = raw.parent / "masks"
    manifest = build_manifest(raw, masks if masks.is_dir() else None, cfg.data.near_duplicate_hamming)
    cfg.data.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(cfg.data.manifest_path, index=False)
    run_eda(manifest, raw, Path(args.reports_dir) / "eda", cfg.preprocess.image_size)
    split = make_split(manifest, cfg.data.val_fraction, cfg.data.test_fraction, cfg.data.seed)
    cfg.data.split_path.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(cfg.data.split_path, index=False)
    summary = split_summary(split)
    (Path(args.reports_dir) / "eda" / "split.md").write_text(
        "# Split (stratified by class/defect type, duplicate-group aware)\n\n" + summary.to_markdown() + "\n",
        encoding="utf-8",
    )
    print(summary.to_string())


def cmd_train(args) -> None:
    from defect_detection.training.trainer import run_training

    print(json.dumps(run_training(_cfg(args)), indent=2))


def cmd_export(args) -> None:
    from defect_detection.export import export_bundle

    print(export_bundle(_cfg(args)))


def cmd_evaluate(args) -> None:
    from defect_detection.evaluation.evaluate import run_evaluation

    m = run_evaluation(_cfg(args), args.split)
    print(json.dumps({k: m[k] for k in ("at_threshold", "average_precision", "roc_auc")}, indent=2))


def cmd_errors(args) -> None:
    from defect_detection.evaluation.error_analysis import run_error_analysis

    print(json.dumps(run_error_analysis(_cfg(args)), indent=2))


def cmd_benchmark(args) -> None:
    from defect_detection.evaluation.benchmark import run_benchmark

    print(json.dumps(run_benchmark(_cfg(args), args.iters, args.threads), indent=2))


def cmd_robustness(args) -> None:
    from defect_detection.evaluation.robustness import run_perturbation_sweep

    print(run_perturbation_sweep(_cfg(args)).to_string(index=False))


def cmd_holdout(args) -> None:
    from defect_detection.evaluation.robustness import run_unseen_defect_holdout

    print(run_unseen_defect_holdout(_cfg(args), args.epochs).to_string(index=False))


def cmd_cv(args) -> None:
    from defect_detection.evaluation.cross_validation import run_cross_validation

    r = run_cross_validation(_cfg(args), args.folds)
    print(json.dumps({k: r[k] for k in ("pooled_oof_at_fold_thresholds", "pooled_ci95_exact", "fold_mean")},
                     indent=2, default=float))


def cmd_pipeline(args) -> None:
    """train -> export (+parity check) -> evaluate -> error analysis -> benchmark."""
    from defect_detection.evaluation.benchmark import run_benchmark
    from defect_detection.evaluation.error_analysis import run_error_analysis
    from defect_detection.evaluation.evaluate import run_evaluation
    from defect_detection.export import export_bundle
    from defect_detection.training.trainer import run_training

    cfg = _cfg(args)
    run_training(cfg)
    export_bundle(cfg)
    run_evaluation(cfg)
    run_error_analysis(cfg)
    run_benchmark(cfg)
    log.info("pipeline complete", extra={"run_dir": str(cfg.run_dir)})


def cmd_compare(args) -> None:
    """Train every backbone on the identical split and tabulate accuracy/latency/size."""
    from defect_detection.evaluation.compare import compare_backbones

    compare_backbones(args.config, args.backbones.split(","), Path(args.reports_dir), args.set or [],
                      skip_existing=args.skip_existing)


def cmd_promote(args) -> None:
    """Copy a run's bundle to the serving location and its reports into reports/."""
    cfg = _cfg(args)
    src, dst = cfg.run_dir / "model", Path(args.model_dir)
    if not (src / "model.onnx").exists():
        sys.exit(f"no exported model in {src}")
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("model.onnx", "metadata.json"):
        shutil.copy2(src / name, dst / name)
    reports = Path(args.reports_dir)
    for sub in ("eval", "error_analysis", "robustness"):
        if (cfg.run_dir / sub).exists():
            shutil.copytree(cfg.run_dir / sub, reports / sub, dirs_exist_ok=True)
    for f in ("training_curves.png", "train_summary.json", "benchmark.json"):
        if (cfg.run_dir / f).exists():
            shutil.copy2(cfg.run_dir / f, reports / f)
    # Experiment-level analyses (cross-validation, unseen-defect hold-out), if they were run.
    extra = ((cfg.artifacts_dir / "cv" / cfg.experiment_name, "cv"), (cfg.artifacts_dir / "holdout", "holdout"))
    for src_dir, name in extra:
        if src_dir.is_dir():
            dst_dir = reports / name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.suffix in (".md", ".png", ".json") or f.name == "oof_predictions.csv":
                    shutil.copy2(f, dst_dir / f.name)
    log.info("promoted", extra={"run": cfg.experiment_name, "model_dir": str(dst)})


def cmd_predict(args) -> None:
    """Offline batch inference over a file or folder -> CSV (or stdout JSON)."""
    import csv

    from defect_detection.inference.predictor import DefectPredictor
    from defect_detection.preprocessing import ImageDecodeError, decode_image

    predictor = DefectPredictor(args.model_dir)
    src = Path(args.input)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.suffix.lower() in exts)
    results = []
    for start in range(0, len(files), args.batch_size):
        chunk, images, ok = files[start : start + args.batch_size], [], []
        for f in chunk:
            try:
                images.append(decode_image(f.read_bytes()))
                ok.append(f)
            except ImageDecodeError as exc:
                results.append({"file": str(f), "error": str(exc)})
        preds, _ = predictor.predict(images)
        results += [{"file": str(f), **p.to_dict()} for f, p in zip(ok, preds, strict=True)]
    if args.output:
        keys = ["file", "predicted_class", "confidence", "defect_probability", "threshold",
                "review_recommended", "error"]
        with open(args.output, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(results)
        log.info("predictions written", extra={"n": len(results), "path": args.output})
    else:
        print(json.dumps(results, indent=2))


def cmd_serve(args) -> None:
    import uvicorn

    uvicorn.run("defect_detection.serving.app:create_app", factory=True, host=args.host, port=args.port,
                workers=args.workers, log_config=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vdd", description="Visual defect detection toolkit")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--json-logs", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def with_cfg(sp):
        sp.add_argument("--config", default="configs/train.yaml")
        sp.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override config, e.g. train.backbone=resnet18")
        return sp

    s = sub.add_parser("download", help="fetch public MVTec AD category")
    s.add_argument("--category", default="metal_nut")
    s.add_argument("--data-dir", default="data")
    s.set_defaults(fn=cmd_download)

    s = with_cfg(sub.add_parser("prepare", help="manifest + EDA + split"))
    s.add_argument("--reports-dir", default="reports")
    s.set_defaults(fn=cmd_prepare)

    for name, fn, help_ in (("train", cmd_train, "train a model"), ("export", cmd_export, "export ONNX bundle"),
                            ("analyze-errors", cmd_errors, "FP/FN + Grad-CAM analysis"),
                            ("robustness", cmd_robustness, "perturbation robustness sweep"),
                            ("pipeline", cmd_pipeline, "train->export->evaluate->errors->benchmark")):
        with_cfg(sub.add_parser(name, help=help_)).set_defaults(fn=fn)

    s = with_cfg(sub.add_parser("evaluate", help="evaluate exported bundle"))
    s.add_argument("--split", default="test", choices=["val", "test"])
    s.set_defaults(fn=cmd_evaluate)

    s = with_cfg(sub.add_parser("benchmark", help="latency benchmark"))
    s.add_argument("--iters", type=int, default=200)
    s.add_argument("--threads", type=int, default=None)
    s.set_defaults(fn=cmd_benchmark)

    s = with_cfg(sub.add_parser("cv", help="k-fold cross-validation with out-of-fold predictions"))
    s.add_argument("--folds", type=int, default=5)
    s.set_defaults(fn=cmd_cv)

    s = with_cfg(sub.add_parser("holdout", help="unseen-defect-type generalisation (re-trains per type)"))
    s.add_argument("--epochs", type=int, default=15)
    s.set_defaults(fn=cmd_holdout)

    s = with_cfg(sub.add_parser("compare", help="compare backbones"))
    s.add_argument("--backbones", default="mobilenet_v3_large,efficientnet_b0,resnet18,resnet50")
    s.add_argument("--reports-dir", default="reports")
    s.add_argument("--skip-existing", action="store_true")
    s.set_defaults(fn=cmd_compare)

    s = with_cfg(sub.add_parser("promote", help="publish a run as the served model"))
    s.add_argument("--model-dir", default="artifacts/model")
    s.add_argument("--reports-dir", default="reports")
    s.set_defaults(fn=cmd_promote)

    s = sub.add_parser("predict", help="offline batch inference")
    s.add_argument("input")
    s.add_argument("--model-dir", default="artifacts/model")
    s.add_argument("--output")
    s.add_argument("--batch-size", type=int, default=16)
    s.set_defaults(fn=cmd_predict)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--workers", type=int, default=1)
    s.set_defaults(fn=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level, args.json_logs)
    args.fn(args)


if __name__ == "__main__":
    main()
