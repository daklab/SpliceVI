# Getting Started

## Installation

### 1. Create a conda environment

```bash
conda create -n splicevi-env python=3.12
conda activate splicevi-env
```

### 2. Clone and install

```bash
git clone https://github.com/daklab/SpliceVI.git
cd SpliceVI
pip install -e .
```

This installs all dependencies from `pyproject.toml`. After installation, `from splicevi import SPLICEVI` works from any script or notebook.

---

## Data format

SpliceVI expects a [MuData](https://mudata.readthedocs.io/) object with two modalities that hold the same cells:

- `rna` holds gene expression `(cells × genes)`. The model reads `layers["length_norm"]`, and the library size of each
  cell is in `mdata.obsm["X_library_size"]`.
- `splicing` holds the junctions `(cells × junctions)`, with these layers:
    - `cell_by_junction_matrix`, the reads of each junction
    - `cell_by_cluster_matrix`, the reads of the junction's ATSE, repeated for each junction of the ATSE
    - `junc_ratio`, the junction usage (PSI), which is junction reads / ATSE reads and 0 where the ATSE has no reads
    - `psi_mask`, which is 1 where the ATSE has reads in the cell and 0 otherwise

  Its `var["event_id"]` gives the ATSE each junction belongs to.

These are the default names of `SPLICEVI.setup_mudata`, and other names can be passed to it. The
[SpliceVI MuData Object](data-format.md) page describes every field. See the [SplicingDataset](https://daklab.github.io/LeafletFA/data-layer/splicingdataset/) documentation in LeafletFA for utilities to construct this format from raw data.

---

## Training

### Using the Python API

```python
import mudata
from splicevi import SPLICEVI

mdata = mudata.read_h5mu("train_data.h5mu")

# Layer and modality names default to the layout in "SpliceVI MuData Object";
# pass e.g. rna_layer="raw_counts" if your expression counts are in another layer.
SPLICEVI.setup_mudata(mdata, batch_key="mouse.id")

# The defaults are the paper's model and training settings.
model = SPLICEVI(mdata)
model.train()

model.save("models/my_run/")
```

### Using the SLURM script

`train_splicevi.py` trains one model from a MuData file. Every argument of `SPLICEVI(...)` and `train()` is also a
command-line flag, and flags left out keep the paper defaults.

```bash
python train_splicevi.py --train_mdata_path train_data.h5mu --model_dir models/my_run --batch_key mouse.id --seed 1
python train_splicevi.py --help     # all flags with their defaults
```

`train_splicevi.sh` runs the same command as a SLURM job. Set the data path, output folder and batch key at its top
(other settings go in its `EXTRA_ARGS` line, for example `EXTRA_ARGS="--max_epochs 100 --lr 5e-4"`) and submit it.

```bash
mkdir -p logs && sbatch train_splicevi.sh
```

---

## Extracting outputs

```python
model = SPLICEVI.load("models/my_run/", adata=mdata)

# Joint latent representation
latent = model.get_latent_representation()          # (cells × n_latent)

# Modality-specific latents
z_expr = model.get_latent_representation(modality="expression")
z_spl  = model.get_latent_representation(modality="splicing")

# Imputed PSI values: posterior mean junction usage given each cell's reads
psi = model.get_normalized_splicing_DM()            # (cells × junctions)
```
