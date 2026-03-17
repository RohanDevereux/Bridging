# Bridging

Protein-protein binding affinity project. The final report uses a single active pipeline:

`PPB-Affinity dataset -> OpenMM MD -> GraphVAE dataset preparation -> GraphVAE training / sweeps / resampling`

Everything outside that path has been moved under [`ARCHIVE/`](ARCHIVE/README.md).

## Active Code

Active code lives in:

- [`src/bridging/MD`](src/bridging/MD)
- [`src/bridging/graphvae`](src/bridging/graphvae)
- [`src/bridging/utils`](src/bridging/utils)
- [`src/bridging/dataProcessing/filter_ppb_broad.py`](src/bridging/dataProcessing/filter_ppb_broad.py)
- [`src/bridging/dataProcessing/preshard_dataset.py`](src/bridging/dataProcessing/preshard_dataset.py)

Active HPC entry points live in:

- [`hpc/md_run_dataset.sbatch`](hpc/md_run_dataset.sbatch)
- [`hpc/md_run_shard_00.sbatch`](hpc/md_run_shard_00.sbatch)
- [`hpc/md_run_shard_01.sbatch`](hpc/md_run_shard_01.sbatch)
- [`hpc/graphvae_prepare_sharded_cpu.sbatch`](hpc/graphvae_prepare_sharded_cpu.sbatch)
- [`hpc/graphvae_merge_prepared_cpu.sbatch`](hpc/graphvae_merge_prepared_cpu.sbatch)
- [`hpc/graphvae_materialize_views_cpu.sbatch`](hpc/graphvae_materialize_views_cpu.sbatch)
- [`hpc/graphvae_saved_sweep_gpu.sbatch`](hpc/graphvae_saved_sweep_gpu.sbatch)
- [`hpc/graphvae_full_gpu.sbatch`](hpc/graphvae_full_gpu.sbatch)
- [`hpc/graphvae_augment_torsions_part_cpu.sbatch`](hpc/graphvae_augment_torsions_part_cpu.sbatch)
- [`hpc/graphvae_augment_force_features_part_cpu.sbatch`](hpc/graphvae_augment_force_features_part_cpu.sbatch)
- [`hpc/graphvae_resample_gpu.sbatch`](hpc/graphvae_resample_gpu.sbatch)
- [`hpc/submit_graphvae_prepare_parallel.sh`](hpc/submit_graphvae_prepare_parallel.sh)
- [`hpc/submit_graphvae_torsions_parallel.sh`](hpc/submit_graphvae_torsions_parallel.sh)
- [`hpc/submit_graphvae_force_features_parallel.sh`](hpc/submit_graphvae_force_features_parallel.sh)
- [`hpc/submit_graphvae_force_split.sh`](hpc/submit_graphvae_force_split.sh)
- [`hpc/submit_graphvae_resample_matrix.sh`](hpc/submit_graphvae_resample_matrix.sh)

## Archived Work

Old approaches and abandoned branches now live under [`ARCHIVE/`](ARCHIVE/README.md):

- [`ARCHIVE/CA_contact_map`](ARCHIVE/CA_contact_map/README.md)
- [`ARCHIVE/MMGBSA`](ARCHIVE/MMGBSA/README.md)
- [`ARCHIVE/PRODIGY`](ARCHIVE/PRODIGY/README.md)
- [`ARCHIVE/PPB_Affinity_augmentation`](ARCHIVE/PPB_Affinity_augmentation/README.md)
- [`ARCHIVE/Data_curation`](ARCHIVE/Data_curation/README.md)

These directories are preserved for traceability and appendix material, but they are not part of the active final-report workflow.

## Active Flow

### 1. Create the main environment

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

### 2. Run MD

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

### 3. Build prepared graph records

This is the main CPU-bound graph construction step. It builds:

- DeepRank2 static node and edge features
- residue identity / physicochemical node features
- MD dynamic node and edge summaries
- torsion node features
- node-level inter-chain Coulomb / LJ force summaries

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

Prepared output:

```bash
src/bridging/generatedData/graphvae/<dataset_stem>_parallel_prepare/prepared/graph_records.pt
```

### 4. Materialize graph views

This step creates the `full` and `interface` record variants used by the saved-graph sweep.

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

sbatch hpc/graphvae_materialize_views_cpu.sbatch
```

### 5. Run GraphVAE sweeps

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

sbatch hpc/graphvae_saved_sweep_gpu.sbatch
```

### 6. Run focused resampling / confirmatory matrix

```bash
cd ~/Bridging
source .venv/bin/activate
export PYTHONPATH=src

bash hpc/submit_graphvae_resample_matrix.sh
```

### 7. Optional backfill for older prepared datasets

These only exist to upgrade older prepared datasets that predate torsion or force features.

Backfill torsions:

```bash
bash hpc/submit_graphvae_torsions_parallel.sh
```

Backfill force features:

```bash
bash hpc/submit_graphvae_force_features_parallel.sh
```

New graph builds should use the complete prepare path from step 3 instead of these backfills.

## Repo Layout

```text
src/bridging/
  MD/              Active MD simulation / preparation code
  graphvae/        Active graph construction, views, training, sweeps, resampling
  utils/           Shared helpers used by the active path
  dataProcessing/  Only the dataset scripts still needed by the active path
  rawData/         Input datasets and source tables
  processedData/   Cleaned / split CSVs used by the active path
  generatedData/   Generated outputs

ARCHIVE/
  ...              Historical approaches not used in the final report
```
