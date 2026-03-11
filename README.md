# Bridging

Protein-protein binding affinity project built around:
- OpenMM MD trajectories
- DeepRank2-derived residue graphs
- GraphVAE models for static (`S`) vs static+dynamics (`SD`)
- optional MM/GBSA baselines and correction models

MD simulation and graph construction are the expensive stages. In practice they should be run on a supercomputer and can take days to weeks for large datasets.

## Quickstart

### 1. Project Python environment
Create the main project environment in the repo root:

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

### 2. AmberTools environment for MM/GBSA
Keep AmberTools separate from `.venv`.

The repo includes:

```bash
environment.ambertools.yml
```

Create that environment with conda or micromamba. On Sonic, a user-local micromamba install under `~/scratch` is the simplest path.

Micromamba bootstrap on Sonic:

```bash
cd ~/scratch
mkdir -p micromamba-bin micromamba

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C micromamba-bin --strip-components=1 bin/micromamba

export MAMBA_ROOT_PREFIX="$HOME/scratch/micromamba"
export PATH="$HOME/scratch/micromamba-bin:$PATH"

micromamba env create -f ~/Bridging/environment.ambertools.yml
```

Expose AmberTools binaries in a shell that already has `.venv` active:

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
```

## Run Order

### 3. Run MD on Sonic
The provided MD sbatch shards the dataset into two GPU array tasks.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export OUT_ROOT="$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20"

sbatch hpc/md_run_dataset.sbatch
```

Outputs:

```bash
$HOME/scratch/MD_datasets/<dataset_stem>/<PDB>/
```

### 4. Build graph records after MD and DeepRank2 HDF5 are ready
This is the CPU-bound prepare stage used in the current GraphVAE workflow.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export DEEPRANK_HDF5="$HOME/scratch/deeprank/PPB_Affinity_broad_pairuniq_train80_test20/graphs_fullcomplex_v1.hdf5"
export N_SHARDS=3
export CPUS_PER_SHARD=16

bash hpc/submit_graphvae_prepare_parallel.sh
```

Merged prepared output:

```bash
src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_parallel_prepare/merged/prepared/graph_records.pt
```

### 5. Add torsion features
This uses the resumable shard-based torsion augmentation path.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export MD_ROOT="$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20"
export OUT_ROOT="src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_input_v1"

bash hpc/submit_graphvae_torsions_parallel.sh
```

Merged torsion-prepared output:

```bash
src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_input_v1/prepared/graph_records.pt
```

### 6. Run MM/GBSA on Sonic
The MM/GBSA launcher supports preset configurations:

- `balanced_gb`
- `full_gb`
- `pb_pilot`
- `full_pb`

For the current project, `full_gb` is the main higher-fidelity setting.

Example:

```bash
cd ~/Bridging
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

Merged MM/GBSA CSV:

```bash
src/bridging/generatedData/MMGBSA/PPB_Affinity_broad_pairuniq_train80_test20_<run_tag>/PPB_Affinity_broad_pairuniq_train80_test20_mmgbsa_estimates.csv
```

Live partial merge during a running MM/GBSA campaign:

```bash
python -m bridging.MMGBSA.merge_sharded_results \
  --shard-root "$OUT_ROOT" \
  --out "$OUT_ROOT/live_partial_mmgbsa.csv"
```

### 7. Train and probe the GraphVAE
The GPU wrapper runs:
- preflight
- `S` and `SD` training
- latent export
- ridge probe
- optional MM/GBSA correction evaluation

Without MM/GBSA:

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export DATASET_STEM="$(basename "$DATASET" .csv)"
export SCRATCH_ROOT="$HOME/scratch"
export MD_ROOT="$SCRATCH_ROOT/MD_datasets/$DATASET_STEM"
export PDB_CACHE_ROOT="$SCRATCH_ROOT/pdb_cache"
export OUT_DIR="src/bridging/generatedData/graphvae/${DATASET_STEM}_torsion_input_v1"
export DEEPRANK_HDF5="$SCRATCH_ROOT/deeprank/$DATASET_STEM/graphs_fullcomplex_v1.hdf5"
export MSMS_BIN_DIR="$HOME/scratch/conda/msms/bin"
export PATH="$MSMS_BIN_DIR:$PATH"

export REUSE_PREPARED=1
export BUILD_DEEPRANK=0
export DEVICE=cuda
export RIDGE_CV_FOLDS=5
export RIDGE_CV_REPEATS=2
export RIDGE_CV_INNER_FOLDS=5

sbatch hpc/graphvae_full_gpu.sbatch
```

With MM/GBSA correction:

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export DATASET_STEM="$(basename "$DATASET" .csv)"
export SCRATCH_ROOT="$HOME/scratch"
export MD_ROOT="$SCRATCH_ROOT/MD_datasets/$DATASET_STEM"
export PDB_CACHE_ROOT="$SCRATCH_ROOT/pdb_cache"
export OUT_DIR="src/bridging/generatedData/graphvae/${DATASET_STEM}_torsion_input_v1"
export DEEPRANK_HDF5="$SCRATCH_ROOT/deeprank/$DATASET_STEM/graphs_fullcomplex_v1.hdf5"
export MSMS_BIN_DIR="$HOME/scratch/conda/msms/bin"
export PATH="$MSMS_BIN_DIR:$PATH"
export MMGBSA_CSV="src/bridging/generatedData/MMGBSA/${DATASET_STEM}_gb_full_done1458/PPB_Affinity_broad_pairuniq_train80_test20_mmgbsa_estimates.csv"

export REUSE_PREPARED=1
export BUILD_DEEPRANK=0
export DEVICE=cuda
export RIDGE_CV_FOLDS=5
export RIDGE_CV_REPEATS=2
export RIDGE_CV_INNER_FOLDS=5

sbatch hpc/graphvae_full_gpu.sbatch
```

## Main outputs

Prepared graphs:

```bash
src/bridging/generatedData/graphvae/<dataset_stem>_parallel_prepare/merged/prepared/graph_records.pt
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_input_v1/prepared/graph_records.pt
```

GraphVAE results:

```bash
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_input_v1/mode_S/
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_input_v1/mode_SD/
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_input_v1/compare_S_vs_SD.json
```

MM/GBSA results:

```bash
src/bridging/generatedData/MMGBSA/<dataset_stem>_<run_tag>/
```
