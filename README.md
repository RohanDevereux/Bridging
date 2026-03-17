# Bridging

Protein-protein binding affinity project.

Active final pipeline:

`PPB-Affinity -> OpenMM MD -> graph preparation -> full/interface GraphVAE or supervised baseline -> resample evaluation`

Everything not used in that final path lives under [`ARCHIVE/`](ARCHIVE/README.md).

## Run

Create the environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
export PYTHONPATH=src
```

Run the active path from scratch:

```bash
sbatch hpc/md_run_dataset.sbatch
bash hpc/submit_graphvae_prepare_parallel.sh
sbatch hpc/graphvae_materialize_views_cpu.sbatch
bash hpc/submit_graphvae_resample_matrix.sh
```

If MD and prepared/view records already exist, start from the last command.

## Where To Look

- [`src/bridging/MD/`](src/bridging/MD)
  Active OpenMM preparation and simulation.
- [`src/bridging/graphvae/`](src/bridging/graphvae/README.md)
  Active graph preparation, models, and runners.
- [`RESULTS/`](RESULTS/README.md)
  Final resample results snapshot committed to the repo.
- [`ARCHIVE/`](ARCHIVE/README.md)
  Old approaches kept for traceability only.
