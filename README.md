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
It also pins `numpy<2` because AmberTools / ParmEd still imports `numpy.compat`.

If Sonic provides a working AmberTools module, that is also acceptable. If the module is incompatible with worker-node CPUs, use a user-local micromamba/conda environment and expose only its binaries.

### Sonic: recommended AmberTools setup
On Sonic, the most reliable setup is:

1. use `.venv` for all `python -m bridging...` commands
2. install AmberTools into a user-local micromamba env under `~/scratch`
3. expose only the AmberTools binaries from that env on `PATH`
4. do not `activate` the micromamba env in your shell

If `conda` is not available on the login node, bootstrap micromamba once:

```bash
cd ~/scratch
mkdir -p micromamba-bin micromamba

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C micromamba-bin --strip-components=1 bin/micromamba

export MAMBA_ROOT_PREFIX="$HOME/scratch/micromamba"
export PATH="$HOME/scratch/micromamba-bin:$PATH"

micromamba env create -f ~/Bridging/environment.ambertools.yml
```

The environment file creates:

```bash
$HOME/scratch/micromamba/envs/bridging-ambertools
```

Do not run `pip install -e .` inside that AmberTools env.
That env is not the main project environment.

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
module unload amber/ambertools25 2>/dev/null || true
unset AMBERTOOLS_MODULE
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
"$AMBERHOME/bin/python" -c "import numpy, sys; print(sys.executable); print(numpy.__version__)"
python -m bridging.MMGBSA.prefetch_dataset --help
```

Important:
- The Python used for `python -m bridging...` should still be `.venv/bin/python`.
- The AmberTools env only needs to contribute `tleap`, `cpptraj`, and `MMPBSA.py` on `PATH`.
- You generally should not run `pip install -e .` inside the AmberTools env.
- If `MMPBSA.py` fails with `numpy.compat` import errors, the AmberTools env has NumPy 2.x. Reinstall `numpy<2` in that env.
- If batch logs show `Loading amber/ambertools25`, the system module is still leaking into the job environment. Clear `AMBERTOOLS_MODULE` and use `AMBERTOOLS_BIN` instead.
- On Sonic, the `amber/ambertools25` module may fail on some worker nodes due to CPU instruction-set mismatch. The user-local micromamba env avoids that.
- The AmberTools env may contain a stale partial editable `bridging` install from earlier experiments. That does not matter as long as `.venv` is the Python actually running `bridging`.

### Repairing a broken AmberTools env
If you accidentally created the AmberTools env with Python 3.13 or NumPy 2.x, either recreate it from `environment.ambertools.yml` or at minimum force NumPy back below 2:

```bash
export MAMBA_ROOT_PREFIX="$HOME/scratch/micromamba"
export AMBERHOME="$MAMBA_ROOT_PREFIX/envs/bridging-ambertools"
"$AMBERHOME/bin/python" -m pip install "numpy<2"
"$AMBERHOME/bin/python" -c "import numpy; print(numpy.__version__)"
```

### MMGBSA Sonic launch presets
`hpc/submit_mmgbsa_parallel.sh` now accepts `MMGBSA_PRESET`:

- `balanced_gb`: production-only GB with frame stride 5
- `full_gb`: production-only GB using every saved production frame
- `pb_pilot`: production-only PB with frame stride 2
- `full_pb`: production-only PB using every saved production frame

Example:

```bash
source .venv/bin/activate
export PYTHONPATH=src
export PYTHONNOUSERSITE=1
module unload amber/ambertools25 2>/dev/null || true
unset AMBERTOOLS_MODULE
export MAMBA_ROOT_PREFIX="$HOME/scratch/micromamba"
export AMBERHOME="$MAMBA_ROOT_PREFIX/envs/bridging-ambertools"
export AMBERTOOLS_BIN="$AMBERHOME/bin"
export PATH="$AMBERTOOLS_BIN:$PATH"

export DATASET="tmp/PPB_Affinity_broad_pairuniq_train80_test20_done1458.csv"
export N_SHARDS=48
export CPUS_PER_SHARD=1
export MMGBSA_PRESET=full_gb
export RUN_TAG=gb_full_done1458

bash hpc/submit_mmgbsa_parallel.sh
```

Notes:
- `balanced_gb` is the cheap baseline.
- `full_gb` is the main "better accuracy, slower" setting.
- `pb_pilot` should be used before `full_pb`; do not launch a full PB campaign blind.
- Each shard CSV is saved after every processed row. Partial results are therefore usable mid-run.
- To build a live partial merged CSV while a run is still active:

```bash
python -m bridging.MMGBSA.merge_sharded_results \
  --shard-root "$OUT_ROOT" \
  --out "$OUT_ROOT/live_partial_mmgbsa.csv"
```

### Resumable torsion augmentation on Sonic
Avoid running `bridging.graphvae.augment_torsions --records-in ...` interactively on the merged file.
That path only writes `graph_records.pt` at the end.

Use the sharded launcher instead:

```bash
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export MD_ROOT="$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20"
export OUT_ROOT="src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_input_v1"

bash hpc/submit_graphvae_torsions_parallel.sh
```

This augments the base and parallel-prepare checkpoint shards separately and then merges them back into:

```bash
src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_input_v1/prepared/graph_records.pt
```

Notes:
- This path is resumable at the checkpoint-directory level.
- If a torsion array task is re-run and the output checkpoint count already matches the input checkpoint count, that part is skipped.
- The merged `graph_records.pt` is only written after the dependency merge job completes.

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
