`graphvae/` contains the active graph-based modelling pipeline.

## Layout

- `common/`
  Shared config and identifiers.
- `prep/`
  Graph construction, MD-derived features, and full/interface view materialisation.
- `ml/`
  Datasets, GraphVAE, supervised baseline, and latent regression.
- `runners/`
  Experiment entry points. The final thesis comparison is driven by `runners/resample_config.py`. `runners/run_full.py` and `runners/sweep.py` are broader alternate runners kept for support and earlier workflow variants, not the primary final thesis route.
- `tools/`
  Audits and maintenance scripts that are not part of the main final path.

## Start Here

If you are reading the core final thesis pipeline, start in this order:

1. `prep/prepare.py`
2. `prep/record_views.py`
3. `ml/train.py`
4. `ml/supervised_baseline.py`
5. `ml/regress.py`
6. `runners/resample_config.py`

