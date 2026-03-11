# Bridging
(In Progress); Thesis Project Work

Working on bridging simulation and experiment for protein-protein binding affinity calculations using deep neural networks

## Environment Setup
Use two environments.

1. Project Python environment: runs the `bridging` codebase.
2. AmberTools environment: provides external MM/GBSA executables only.

Do not treat the AmberTools environment as the main project environment.

### Project environment (`.venv`)
Create `.venv` with Python 3.11 or 3.12, then install the repo from `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Quick check:

```bash
source .venv/bin/activate
python -c "from Bio.PDB import PDBParser; import h5py; import mdtraj; print('project env ok')"
```

Notes:
- `pip install -e .` should be run in `.venv`, not in the AmberTools env.
- If `.venv` predates dependency changes in `pyproject.toml`, rerun `python -m pip install -e .`.
- `PYTHONPATH=src` is not required after a successful editable install, but it is harmless.

## AmberTools for MMGBSA
MMGBSA in this repo uses external AmberTools executables:
`tleap`, `cpptraj`, `MMPBSA.py`.

Keep your project Python in `.venv`, and install AmberTools separately with conda/micromamba:

```bash
conda env create -f environment.ambertools.yml
```

`environment.ambertools.yml` pins Python 3.12 to avoid landing in Python 3.13, which does not satisfy this repo's `pyproject.toml` requirement (`>=3.10,<3.13`).

If Sonic provides a working AmberTools module, that is also acceptable. If the module is incompatible with worker-node CPUs, use a user-local micromamba/conda environment and expose only its binaries.

Then, in `bash`, activate your project venv and expose AmberTools binaries on `PATH`:

```bash
source .venv/bin/activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bridging-ambertools
export AMBERTOOLS_BIN="$CONDA_PREFIX/bin"
conda deactivate
export PATH="$AMBERTOOLS_BIN:$PATH"
```

For a user-local micromamba install, the equivalent is:

```bash
source .venv/bin/activate
export PYTHONNOUSERSITE=1
export MAMBA_ROOT_PREFIX="$HOME/scratch/micromamba"
export AMBERHOME="$MAMBA_ROOT_PREFIX/envs/bridging-ambertools"
export AMBERTOOLS_BIN="$AMBERHOME/bin"
export PATH="$AMBERTOOLS_BIN:$PATH"
```

Quick check:

```bash
which tleap cpptraj MMPBSA.py
tleap -h >/dev/null && echo "tleap ok"
cpptraj -h >/dev/null && echo "cpptraj ok"
MMPBSA.py -h >/dev/null && echo "MMPBSA.py ok"
python -m bridging.MMGBSA.prefetch_dataset --help
```

Important:
- The Python used for `python -m bridging...` should still be `.venv/bin/python`.
- The AmberTools env only needs to contribute `tleap`, `cpptraj`, and `MMPBSA.py` on `PATH`.
- You generally should not run `pip install -e .` inside the AmberTools env.

## GraphVAE (S vs SD) Pipeline
New pipeline in `src/bridging/graphvae/`:
- `S`: static structure features only
- `SD`: static + MD dynamic features
- shared 8-D masked Graph-VAE encoder, then linear Ridge probe on latent `mu_0..mu_7`

Install extras:

```bash
pip install -e .
```

Notes:
- `deeprank2` and `torch-geometric` are required dependencies for this pipeline.
- Default DeepRank influence radius is set very large (`1e6`) to approximate whole-complex node coverage.
- By default, prep fails if any protein residue (with CA) is missing from the DeepRank node set. Override with `--allow-partial-node-coverage`.
- DeepRank2 upstream currently documents Python 3.10 support; if your main environment is Python 3.11, generate HDF5 in a Py3.10 env and train/probe with existing files.
- Default `--graph-source` is `md_topology_protein` so DeepRank node IDs match the saved MD topology/trajectory ID space.
- Prep logs include progress/ETA; tune frequency with `--progress-every` (`prepare`) or `--prepare-progress-every` (`run_full`).

Prepare records:

```bash
python -m bridging.graphvae.prepare \
  --dataset src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv \
  --md-root "$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20" \
  --out-dir src/bridging/generatedData/graphvae/ppb_broad_prepared \
  --graph-source md_topology_protein \
  --deep-rank-hdf5 /path/to/deeprank_graphs.hdf5
```

Train mode `S` and export latents:

```bash
python -m bridging.graphvae.train \
  --records src/bridging/generatedData/graphvae/ppb_broad_prepared/graph_records.pt \
  --out-dir src/bridging/generatedData/graphvae/ppb_broad_prepared/mode_S \
  --mode S \
  --latent-dim 8 \
  --device cpu
```

Linear probe:

```bash
python -m bridging.graphvae.regress \
  --latents-csv src/bridging/generatedData/graphvae/ppb_broad_prepared/mode_S/latents_S.csv \
  --out-dir src/bridging/generatedData/graphvae/ppb_broad_prepared/mode_S
```

Full end-to-end run (S + SD + compare):

```bash
python -m bridging.graphvae.run_full \
  --dataset src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv \
  --md-root "$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20" \
  --out-dir src/bridging/generatedData/graphvae/ppb_broad_full \
  --graph-source md_topology_protein \
  --deep-rank-hdf5 /path/to/deeprank_graphs.hdf5 \
  --latent-dim 8 \
  --device cuda
```

GPU sbatch wrapper:

```bash
DATASET=src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv \
MD_ROOT=$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20 \
DEEPRANK_HDF5="/path/to/deeprank_graphs.hdf5" \
sbatch hpc/graphvae_full_gpu.sbatch
```

Dynamic feature variation report (after prepare/full run):

```bash
python -m bridging.graphvae.analyze_dynamic_variation \
  --records src/bridging/generatedData/graphvae/ppb_broad_full/prepared/graph_records.pt \
  --out-dir src/bridging/generatedData/graphvae/ppb_broad_full/dynamic_variation
```
