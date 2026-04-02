# production-ML-system

End-to-end ML system: time-aware features, shared training/serving logic, and consistency checks.

## Install

```bash
cd production-ML-system
python -m venv .venv && source .venv/bin/activate
pip install -e ".[xgboost]"
```

## Quick run

```bash
python -m ml_system.cli demo-data --root ./data
python -m ml_system.cli materialize --root ./data
python -m ml_system.cli refresh-online --root ./data
python -m ml_system.cli train --root ./data
python -m ml_system.cli batch-predict --root ./data
python -m ml_system.cli consistency-check --root ./data
python -m ml_system.cli metrics-append-demo --root ./data
# python -m ml_system.cli serve --root ./data
```

Drift summary (two batch prediction Parquet files): `python -m ml_system.cli drift-summarize --reference a.parquet --current b.parquet`
