`MD/` contains the active OpenMM preparation and simulation pipeline used in the thesis.

## Main Files

- `run_dataset.py`
  Main dataset-level entry point for preparation and simulation.
- `prepare_complex.py`
  Structure preparation, solvation, and system setup.
- `simulate.py`
  OpenMM minimisation, equilibration, and production simulation.
- `config.py`
  Default simulation settings and force-field choices.
- `paths.py`
  Default dataset and cache paths.

## Notes

- The staged thesis route uses `hpc/md_run_dataset.sbatch`.
- `prefetch_dataset.py` and `prefetch_pdbs.py` are support utilities, not the main simulation path.
