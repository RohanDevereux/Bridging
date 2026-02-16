# Bridging
(In Progress); Thesis Project Work

Working on bridging simulation and experiment for protein-protein binding affinity calculations using deep neural networks

## AmberTools for MMGBSA
MMGBSA in this repo uses external AmberTools executables:
`tleap`, `cpptraj`, `MMPBSA.py`.

Keep your project Python in `.venv`, and install AmberTools separately with conda:

```bash
conda env create -f environment.ambertools.yml
```

Then, in `bash`, activate your project venv and expose AmberTools binaries on `PATH`:

```bash
source .venv/bin/activate
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate bridging-ambertools
export AMBERTOOLS_BIN="$CONDA_PREFIX/bin"
conda deactivate
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
