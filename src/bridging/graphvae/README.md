Active package layout for the final MD -> GraphVAE pipeline.

- `common/`
  Shared configuration and utilities used across preparation and modeling.
  Main files: `config.py`, `ids.py`, `splits.py`, `chain_remap.py`, `baselines.py`

- `prep/`
  Scientifically central data-preparation logic.
  Main files: `prepare.py`, `record_views.py`, `deeprank_adapter.py`, `md_dynamics.py`, `force_features.py`

- `ml/`
  Modeling and evaluation logic.
  Main files: `dataset.py`, `model.py`, `train.py`, `supervised_baseline.py`, `regress.py`, `crossval.py`

- `runners/`
  Experiment runners that wire preparation and modeling together.
  Main files: `run_full.py`, `sweep.py`, `resample_config.py`, `materialize_views.py`

- `tools/`
  Diagnostics, audits, and maintenance scripts that are not part of the core final pipeline.
  Main files: `preflight.py`, `analyze_dynamic_variation.py`, `audit_force_compatibility.py`, `augment_*`, `merge_prepared_shards.py`, `build_remaining_dataset.py`
