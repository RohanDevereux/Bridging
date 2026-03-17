# Results

This directory contains a committed snapshot of the final confirmatory run.

Run described here:

- matched retained set: `1437` complexes
- models: `52` total
- matrix: `48` VAE configs + `4` supervised baselines
- evaluation: `5` outer folds x `2` repeats = `10` retrains per config

## Headline Results

- Best overall model: `full_base_S`
  - test RMSE `2.338`
  - test Pearson `0.506`
  - test R^2 `0.215`
- Best VAE: `full_S_semi_shared_static_z32`
  - test RMSE `2.429`
  - test Pearson `0.465`
  - test R^2 `0.154`
- `full` beats `interface` in all `24/24` matched VAE comparisons.
- `S` beats `SD` in `11/12` matched `shared_static` comparisons.
- The current dynamic feature construction did not produce robust gains over the static representation.

## Files

- [`final_resample_matrix_summary.csv`](final_resample_matrix_summary.csv)
  One row per final config, sorted by mean test RMSE. Includes mean, std, min, and max for RMSE, R^2, Pearson `r`, and the train-test / train-val RMSE gaps.

The raw `resample_summary.json` and `resample_fold_metrics.csv` files are not committed because they are bulky and were produced on HPC scratch storage. This CSV is the portable snapshot of the full final matrix.
