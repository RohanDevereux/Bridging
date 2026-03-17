`graphvae/` contains the active graph-based modeling pipeline.

## Layout

- `common/`
  Shared config and identifiers.
- `prep/`
  Graph construction, MD-derived features, and full/interface view materialization.
- `ml/`
  Datasets, GraphVAE, supervised baseline, and latent regression.
- `runners/`
  End-to-end experiment entry points.
- `tools/`
  Audits and maintenance scripts that are not part of the main final path.

## Start Here

If you are reading the core pipeline, start in this order:

1. `prep/prepare.py`
2. `prep/record_views.py`
3. `ml/train.py`
4. `ml/supervised_baseline.py`
5. `ml/regress.py`
6. `runners/resample_config.py`
