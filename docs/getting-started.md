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

SpliceVI expects a [MuData](https://mudata.readthedocs.io/) object with two modalities:

- `rna` — raw gene expression counts `(cells × genes)`
- `splice` — junction usage ratios (PSI values) `(cells × junctions)`

The splicing modality also requires:

- `junc_counts` — integer junction read counts `(cells × junctions)`
- `atse_counts` — total read counts at the ATSE level `(cells × junctions)`
- `psi_observed_mask` — binary mask indicating which junctions are observed per cell

See the [SplicingDataset](https://daklab.github.io/LeafletFA/data-layer/splicingdataset/) documentation in LeafletFA for utilities to construct this format from raw data.

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

Configure paths and hyperparameters at the top of `train_splicevi.sh`, then:

```bash
sbatch train_splicevi.sh
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

# Imputed PSI values
psi = model.get_normalized_splicing()               # (cells × junctions)
```
