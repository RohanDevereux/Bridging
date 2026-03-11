# Bridging

Protein-protein binding affinity project built around:
- OpenMM MD trajectories
- DeepRank2-derived residue graphs
- GraphVAE models for static (`S`) vs static+dynamics (`SD`)
- optional MM/GBSA baselines and correction models

MD simulation and graph construction are the slow stages. For real datasets they should be run on a supercomputer and can take days to weeks.

## Quickstart

### 1. Main project environment
Create the main Python environment in the repo root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Quick check:

```bash
source .venv/bin/activate
python -c "from Bio.PDB import PDBParser; import h5py, mdtraj, torch; print('project env ok')"
```

### 2. AmberTools environment for MM/GBSA
Keep AmberTools separate from `.venv`.

The repo includes:

```bash
environment.ambertools.yml
```

On Sonic, a user-local micromamba install under `~/scratch` works well.

```bash
cd ~/scratch
mkdir -p micromamba-bin micromamba

curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
  | tar -xvj -C micromamba-bin --strip-components=1 bin/micromamba

export MAMBA_ROOT_PREFIX="$HOME/scratch/micromamba"
export PATH="$HOME/scratch/micromamba-bin:$PATH"

micromamba env create -f ~/Bridging/environment.ambertools.yml
```

Use the AmberTools binaries from that env in a shell that already has `.venv` active:

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
```

Quick check:

```bash
which tleap cpptraj MMPBSA.py
"$AMBERHOME/bin/python" -c "import numpy, sys; print(sys.executable); print(numpy.__version__)"
```

## Run order

### 3. Run MD
The MD wrapper shards the dataset across GPU jobs.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export OUT_ROOT="$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20"

sbatch hpc/md_run_dataset.sbatch
```

MD outputs are written under:

```bash
$HOME/scratch/MD_datasets/<dataset_stem>/<PDB>/
```

### 4. Build prepared graph records
This is the CPU-bound DeepRank2 + dynamic-feature prepare stage.

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

Merged output:

```bash
src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_parallel_prepare/merged/prepared/graph_records.pt
```

### 5. Add torsion node features
This appends `sin/cos(phi)` and `sin/cos(psi)` means as input-only node features.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export MD_ROOT="$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20"
export TORSION_OUT_ROOT="src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_input_v1"

bash hpc/submit_graphvae_torsions_parallel.sh
```

Merged output:

```bash
src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_input_v1/prepared/graph_records.pt
```

### 6. Add inter-chain force node features
This appends node-level inter-chain Coulomb/LJ force statistics as input-only node features:
- `dyn_coul_force_mean`
- `dyn_coul_force_std`
- `dyn_lj_force_mean`
- `dyn_lj_force_std`

Run this after the torsion-prepared dataset is available.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export MD_ROOT="$HOME/scratch/MD_datasets/PPB_Affinity_broad_pairuniq_train80_test20"
export PDB_CACHE_ROOT="$HOME/scratch/pdb_cache"
export FORCE_FEATURE_OUT_ROOT="src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_force_input_v1"

bash hpc/submit_graphvae_force_features_parallel.sh
```

Merged output:

```bash
src/bridging/generatedData/graphvae/PPB_Affinity_broad_pairuniq_train80_test20_torsion_force_input_v1/prepared/graph_records.pt
```

### 7. Run MM/GBSA
The MM/GBSA launcher supports these presets:
- `balanced_gb`
- `full_gb`
- `pb_pilot`
- `full_pb`

`full_gb` is the main higher-fidelity setting. It uses production frames and the saved full trajectory as the source, then derives the protein-chain subset needed by MMPBSA.

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
export RUN_TAG=gb_full_done1458_r1
export MMGBSA_OUT_ROOT="src/bridging/generatedData/MMGBSA/PPB_Affinity_broad_pairuniq_train80_test20_${RUN_TAG}"

bash hpc/submit_mmgbsa_parallel.sh
```

Merged output:

```bash
src/bridging/generatedData/MMGBSA/<dataset_stem>_<run_tag>/<dataset_stem>_mmgbsa_estimates.csv
```

Live partial merge during a running MM/GBSA campaign:

```bash
python -m bridging.MMGBSA.merge_sharded_results \
  --shard-root "$MMGBSA_OUT_ROOT" \
  --out "$MMGBSA_OUT_ROOT/live_partial_mmgbsa.csv"
```

### 8. Train and probe the GraphVAE
Choose the prepared dataset you want to train from. The most feature-rich path is the force-augmented one from step 6.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

export DATASET="src/bridging/processedData/PPB_Affinity_broad_pairuniq_train80_test20.csv"
export DATASET_STEM="$(basename "$DATASET" .csv)"
export SCRATCH_ROOT="$HOME/scratch"
export MD_ROOT="$SCRATCH_ROOT/MD_datasets/$DATASET_STEM"
export PDB_CACHE_ROOT="$SCRATCH_ROOT/pdb_cache"
export OUT_DIR="src/bridging/generatedData/graphvae/${DATASET_STEM}_torsion_force_input_v1"
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

With MM/GBSA correction evaluation:

```bash
export MMGBSA_CSV="src/bridging/generatedData/MMGBSA/${DATASET_STEM}_gb_full_done1458_r1/${DATASET_STEM}_mmgbsa_estimates.csv"
sbatch hpc/graphvae_full_gpu.sbatch
```

## Main outputs

Prepared graphs:

```bash
src/bridging/generatedData/graphvae/<dataset_stem>_parallel_prepare/merged/prepared/graph_records.pt
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_input_v1/prepared/graph_records.pt
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_force_input_v1/prepared/graph_records.pt
```

GraphVAE results:

```bash
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_force_input_v1/mode_S/
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_force_input_v1/mode_SD/
src/bridging/generatedData/graphvae/<dataset_stem>_torsion_force_input_v1/compare_S_vs_SD.json
```

MM/GBSA results:

```bash
src/bridging/generatedData/MMGBSA/<dataset_stem>_<run_tag>/
```
