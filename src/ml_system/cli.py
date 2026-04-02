from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ml_system.batch_inference.scorer import BatchScorer
from ml_system.events.bronze import BronzeStore
from ml_system.events.schema import RawEvent
from ml_system.feature_store.offline import OfflineFeatureStore
from ml_system.feature_store.online import OnlineFeatureStore
from ml_system.features.definitions import FEATURE_NAMES
from ml_system.features.engine import FeatureEngine
from ml_system.model_store.registry import ModelRegistry
from ml_system.model_store.bundle import load_bundle
from ml_system.monitoring.metrics import append_prediction_metrics
from ml_system.monitoring.consistency_job import report_to_dict, run_batch_online_consistency
from ml_system.monitoring.drift import summarize_feature_drift
from ml_system.monitoring.metrics import prediction_distribution_shift
from ml_system.training.pipeline import TrainingPipeline
from ml_system.training.snapshot import attach_labels, build_offline_snapshot_from_bronze

try:
    from ml_system.serving.api import create_app
except ImportError:
    create_app = None


def _seed_demo_events(bronze: BronzeStore) -> None:
    base = datetime(2024, 3, 1, tzinfo=timezone.utc)
    ev: list[RawEvent] = []
    for i in range(40):
        t = base + timedelta(days=i // 3, hours=i % 5)
        eid = f"u{i % 4}"
        typ = "click" if i % 2 == 0 else "interaction"
        val = float(1 + (i % 7))
        ev.append(
            RawEvent(
                event_id=f"evt-{i}",
                entity_id=eid,
                event_type=typ,
                occurred_at=t,
                ingested_at=t,
                payload={"value": val},
            )
        )
    bronze.append(ev)


def _demo_labels(features_df):
    import numpy as np
    import pandas as pd

    rows = []
    for _, r in features_df.iterrows():
        x = np.array([r[c] for c in FEATURE_NAMES], dtype=float)
        if np.any(np.isnan(x)):
            y = 0
        else:
            y = int((x[1] + x[3] * 0.01) % 2)
        rows.append(
            {
                "entity_id": r["entity_id"],
                "as_of_ts": r["as_of_ts"],
                "label": y,
            }
        )
    return pd.DataFrame(rows)


def cmd_demo_data(args: argparse.Namespace) -> None:
    root = Path(args.root)
    bronze = BronzeStore(root / "bronze")
    _seed_demo_events(bronze)
    print("Bronze written to", root / "bronze")


def cmd_materialize(args: argparse.Namespace) -> None:
    root = Path(args.root)
    bronze = BronzeStore(root / "bronze")
    offline = OfflineFeatureStore(root / "offline")
    entities = sorted(bronze.read_dataframe()["entity_id"].unique().tolist())
    as_of_times = [
        datetime(2024, 3, 5, tzinfo=timezone.utc),
        datetime(2024, 3, 15, tzinfo=timezone.utc),
        datetime(2024, 3, 25, tzinfo=timezone.utc),
    ]
    job_id, df = build_offline_snapshot_from_bronze(
        bronze, offline, entities=entities, as_of_times=as_of_times
    )
    print("materialization_job_id", job_id, "rows", len(df))


def cmd_refresh_online(args: argparse.Namespace) -> None:
    root = Path(args.root)
    bronze = BronzeStore(root / "bronze")
    online = OnlineFeatureStore(root / "online.db")
    engine = FeatureEngine()
    far = datetime(9999, 1, 1, tzinfo=timezone.utc)
    for eid in sorted(bronze.read_dataframe()["entity_id"].unique().tolist()):
        events = bronze.events_for_entity_up_to(eid, far)
        if not events:
            continue
        ts = max(ev.occurred_at for ev in events)
        fv = engine.compute(eid, ts, events)
        online.upsert_from_values(eid, fv.values)
    print("Online store refreshed")


def cmd_train(args: argparse.Namespace) -> None:
    root = Path(args.root)
    offline = OfflineFeatureStore(root / "offline")
    manifests = sorted((root / "offline").glob("*.meta.json"))
    if not manifests:
        raise SystemExit("run materialize first")
    import json

    meta = json.loads(manifests[-1].read_text())
    job = meta["materialization_job_id"]
    df = offline.read(str(job))
    labels = _demo_labels(df)
    labeled = attach_labels(df, labels)
    train_end = datetime(2024, 3, 10, tzinfo=timezone.utc)
    val_end = datetime(2024, 3, 20, tzinfo=timezone.utc)
    pipe = TrainingPipeline(
        model_store_root=root / "models",
        registry_root=root / "registry",
    )
    bundle = pipe.run(
        labeled,
        train_end=train_end,
        val_end=val_end,
        manifest_out=root / "dataset_manifest.json",
    )
    print("Trained model_version", bundle.model_version)


def cmd_batch_predict(args: argparse.Namespace) -> None:
    root = Path(args.root)
    bronze = BronzeStore(root / "bronze")
    reg = ModelRegistry(root / "registry")
    mv = args.model_version or reg.get_production_version()
    scorer = BatchScorer.from_registry(bronze, model_store=root / "models", model_version=str(mv))
    import pandas as pd

    offline = OfflineFeatureStore(root / "offline")
    manifests = sorted((root / "offline").glob("*.parquet"))
    keys = pd.read_parquet(manifests[-1])[["entity_id", "as_of_ts"]]
    out = scorer.score_dataframe(keys)
    p = scorer.write_parquet(out, root / "batch_out")
    print("Wrote", p)


def cmd_metrics(args: argparse.Namespace) -> None:
    root = Path(args.root)
    import numpy as np
    import pandas as pd

    preds = sorted((root / "batch_out").glob("batch_predictions_*.parquet"))[-1]
    df = pd.read_parquet(preds)
    rng = np.random.default_rng(42)
    y_true = (rng.random(len(df)) > 0.45).astype(int)
    append_prediction_metrics(
        root / "monitoring" / "metrics.jsonl",
        period=args.period,
        y_true=y_true,
        y_pred=df["prediction"].to_numpy(),
    )


def cmd_consistency_check(args: argparse.Namespace) -> None:
    import json

    root = Path(args.root)
    bronze = BronzeStore(root / "bronze")
    online = OnlineFeatureStore(root / "online.db")
    reg = ModelRegistry(root / "registry")
    mv = args.model_version or reg.get_production_version()
    if not mv:
        raise SystemExit("no model in registry")
    bundle = load_bundle(root / "models", str(mv))
    pred_files = sorted((root / "batch_out").glob("batch_predictions_*.parquet"))
    if not pred_files:
        raise SystemExit("no batch predictions; run batch-predict first")
    path = pred_files[-1]
    rpt = run_batch_online_consistency(
        path,
        bronze,
        online,
        bundle,
        epsilon_feat=args.epsilon_feat,
        epsilon_prob=args.epsilon_prob,
    )
    out = report_to_dict(rpt)
    print(json.dumps(out, indent=2, default=str))
    if args.report_out:
        Path(args.report_out).write_text(json.dumps(out, indent=2, default=str))


def cmd_drift_summarize(args: argparse.Namespace) -> None:
    import json

    import pandas as pd

    from ml_system.features.definitions import FEATURE_NAMES

    ref = pd.read_parquet(args.reference)
    cur = pd.read_parquet(args.current)
    rename_cols = {f"feat_{k}": k for k in FEATURE_NAMES}
    ref_f = ref[[c for c in rename_cols if c in ref.columns and c in cur.columns]]
    cur_f = cur[ref_f.columns]
    alt = summarize_feature_drift(
        ref_f.rename(columns=rename_cols),
        cur_f.rename(columns=rename_cols),
        list(rename_cols.values()),
    )
    pred_ref = ref["proba_pos"].to_numpy() if "proba_pos" in ref.columns else ref["prediction"].to_numpy(dtype=float)
    pred_cur = cur["proba_pos"].to_numpy() if "proba_pos" in cur.columns else cur["prediction"].to_numpy(dtype=float)
    pred_shift = prediction_distribution_shift(pred_ref, pred_cur)
    print(json.dumps({"feature_drift": alt, "prediction_drift": pred_shift}, indent=2, default=str))


def cmd_serve(args: argparse.Namespace) -> None:
    if create_app is None:
        raise SystemExit("FastAPI not available")
    import uvicorn

    root = Path(args.root)
    app = create_app(
        bronze_path=root / "bronze",
        online_db=root / "online.db",
        model_store=root / "models",
        registry_path=root / "registry",
    )
    uvicorn.run(app, host=args.host, port=args.port)


def main() -> None:
    p = argparse.ArgumentParser(prog="mlsys")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo-data")
    d.add_argument("--root", default="./data")
    d.set_defaults(func=cmd_demo_data)

    m = sub.add_parser("materialize")
    m.add_argument("--root", default="./data")
    m.set_defaults(func=cmd_materialize)

    o = sub.add_parser("refresh-online")
    o.add_argument("--root", default="./data")
    o.set_defaults(func=cmd_refresh_online)

    t = sub.add_parser("train")
    t.add_argument("--root", default="./data")
    t.set_defaults(func=cmd_train)

    b = sub.add_parser("batch-predict")
    b.add_argument("--root", default="./data")
    b.add_argument("--model-version", default=None)
    b.set_defaults(func=cmd_batch_predict)

    me = sub.add_parser("metrics-append-demo")
    me.add_argument("--root", default="./data")
    me.add_argument("--period", default="2024-03")
    me.set_defaults(func=cmd_metrics)

    cc = sub.add_parser("consistency-check")
    cc.add_argument("--root", default="./data")
    cc.add_argument("--model-version", default=None)
    cc.add_argument("--epsilon-feat", type=float, default=1e-6)
    cc.add_argument("--epsilon-prob", type=float, default=1e-4)
    cc.add_argument("--report-out", default=None)
    cc.set_defaults(func=cmd_consistency_check)

    dr = sub.add_parser("drift-summarize")
    dr.add_argument("--reference", required=True)
    dr.add_argument("--current", required=True)
    dr.set_defaults(func=cmd_drift_summarize)

    s = sub.add_parser("serve")
    s.add_argument("--root", default="./data")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
