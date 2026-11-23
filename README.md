# SpliceVI

[![tests](https://github.com/daklab/SpliceVI/actions/workflows/tests.yml/badge.svg)](https://github.com/daklab/SpliceVI/actions/workflows/tests.yml)

SpliceVI is a variational autoencoder that models single-cell gene expression and alternative splicing together. It
learns one latent space from both measurements and handles the high sparsity of single-cell splicing data: junction
usage is modelled with a Dirichlet-multinomial likelihood over the junctions of each alternative transcript structure
event (ATSE), and the splicing encoder only reads the junctions observed in each cell. With a trained model you can

- embed cells in a joint, expression-only or splicing-only latent space,
- impute junction usage (PSI) and expression for every cell, including junctions without reads,
- test for differential splicing and differential expression between groups of cells, and
- transfer a model trained on a deep atlas to a sparse one by fine-tuning its decoders.

SpliceVI is built on [scvi-tools](https://scvi-tools.org) and follows its API (`setup_mudata`, `train`,
`get_latent_representation`, `differential_expression`, ...).

Documentation: <https://daklab.github.io/SpliceVI/>

Paper: Vaidyanathan S\*, Isaev K\*, Zweig A, Knowles DA. bioRxiv 2025, <https://doi.org/10.1101/2025.11.26.690853>
(\*equal contribution)

---

## Installation

SpliceVI needs Python 3.11 or newer and PyTorch. Install the PyTorch build for your machine first
(<https://pytorch.org/get-started/locally/>), then:

```bash
git clone https://github.com/daklab/SpliceVI.git
cd SpliceVI
pip install -e .
```

To get the exact package versions the paper's models were trained with (PyTorch 2.4.1 for CUDA 12.1, scvi-tools 1.4.1):

```bash
conda env create -f environment.yml
conda activate splicevi-env
```

Optional extras: `pip install -e ".[test]"` (pytest), `".[tutorials]"` (scanpy, matplotlib, jupyter), `".[wandb]"`
(Weights & Biases logging in `train_splicevi.py`).

---

## Quick start

```python
import mudata
from splicevi import SPLICEVI

mdata = mudata.read_h5mu("train_data.h5mu")

# Register the data. The layer and modality names default to the layout described under "Input data" below.
SPLICEVI.setup_mudata(mdata, batch_key="mouse.id")   # batch_key=None if there is no batch to correct for

# The default model and training settings are the ones used in the paper.
model = SPLICEVI(mdata)
model.train()
model.save("models/my_model")

# Latent spaces (cells x 30)
latent_joint = model.get_latent_representation()                          # both modalities
latent_splicing = model.get_latent_representation(modality="splicing")    # or "expression"

# Imputed junction usage (cells x junctions): posterior mean PSI given each cell's reads
psi = model.get_normalized_splicing_DM()

# Differential splicing and expression between two groups of cells
ds = model.differential_splicing(groupby="cell_type", group1="Neuron", group2="Astrocyte", seed=0)
de = model.differential_expression(groupby="cell_type", group1="Neuron", group2="Astrocyte", seed=0)
ds_calls = ds[ds["is_ds_fdr"]]          # junctions called at 5% FDR (|delta PSI| > 0.1)
de_calls = de[de["is_de_fdr_0.05"]]     # genes called at 5% FDR
```

A saved model is loaded with `SPLICEVI.load("models/my_model", adata=mdata)`.

---

## Input data

SpliceVI reads a [MuData](https://mudata.readthedocs.io/) object with a gene expression modality `"rna"` and a splicing
modality `"splicing"`, holding the same cells. The default layer names are:

| Modality | Slot | Contents |
|---|---|---|
| `rna` | `layers["length_norm"]` | expression counts used by the model (raw counts rescaled by transcript length) |
| (MuData) | `obsm["X_library_size"]` | library size of each cell |
| `splicing` | `layers["cell_by_junction_matrix"]` | reads of each junction |
| `splicing` | `layers["cell_by_cluster_matrix"]` | reads of the junction's ATSE (the sum over its junctions), repeated for each junction of the ATSE |
| `splicing` | `layers["junc_ratio"]` | junction usage (PSI): junction reads / ATSE reads, 0 where the ATSE has no reads |
| `splicing` | `layers["psi_mask"]` | 1 where the ATSE has reads in the cell, 0 otherwise |
| `splicing` | `var["event_id"]` | the ATSE each junction belongs to |

Other names can be passed to `setup_mudata`. The full description, including how the splicing layers are built from
junction read counts, is on the [data format page](https://daklab.github.io/SpliceVI/data-format/).

---

## Main settings

- **Defaults.** `SPLICEVI(...)`, `train()`, `differential_splicing()` and `differential_expression()` default to the
  paper's settings. `SPLICEVI.preset("paper", atlas="tms")` returns them as dictionaries, with the variant used for the
  Tabula Muris Senis models (batch correction over mice); `SPLICEVI.preset("sparse")` gives the settings of the models
  for sparse data.
- **Differential splicing** compares junction usage between the groups. Each junction is scored by the posterior
  probability that ΔPSI is at least `delta` (default 0.1) in one direction, and junctions are called at a posterior
  expected false discovery rate of `fdr_target` (default 0.05). By default
  each Monte Carlo sample draws every cell's PSI from its Dirichlet-multinomial posterior
  (`norm_splicing_function="dm_posterior_sample"`). `n_obs_group1` / `n_obs_group2` count the cells with reads for the
  junction; `splicevi.filter_read_support(ds, min_units=3)` keeps junctions with reads in at least 3 cells (or
  metacells) of each group, as in the paper.
- **Differential expression** compares the expected expression under the zero-inflated negative binomial
  (`norm_expression_function="zinb_expected"`). The dropout probability can absorb part of a difference between groups,
  and this setting largely corrects the fold changes that the decoder output alone would shrink.
- **Repeatable results.** Pass `seed=` to `differential_splicing` / `differential_expression` for the same table on
  every call. Setting `scvi.settings.seed` before building the model makes training repeatable on the same machine
  (results on a GPU can still differ slightly between runs).
- **Transfer to a sparse atlas.** `model.finetune_decoders(query_mdata, reference_mdata)` adapts a trained model to new
  data by training only its decoders (and a technology covariate) while keeping the latent space fixed.

---

## Training from the command line

`train_splicevi.py` trains one model from a MuData file and saves it:

```bash
python train_splicevi.py --train_mdata_path train_data.h5mu --model_dir models/my_model --batch_key mouse.id --seed 1
```

Every argument of `SPLICEVI(...)` and `train()` is also a flag (`python train_splicevi.py --help`), and flags left out
keep the paper defaults. `train_splicevi.sh` is a SLURM job script for the same command.

---

## Tests

```bash
pip install -e ".[test]"
pytest
```

The tests build their own small random data and run on CPU in about a minute. Set `SPLICEVI_MODELS` to a folder with
the paper checkpoints to also check them against the presets.

---

## Repository layout

```
SpliceVI/
├── src/splicevi/
│   ├── splicevi.py        # SPLICEVI model class: setup, training, latent spaces, imputation, DE and DS
│   ├── splicevae.py       # the neural network: encoders, decoders, likelihoods
│   ├── presets.py         # the paper's settings as dictionaries; read-support filter for DS results
│   └── finetune.py        # decoder fine-tuning for transfer to a new atlas
├── train_splicevi.py      # command-line training
├── train_splicevi.sh      # SLURM job script for train_splicevi.py
├── tests/                 # pytest suite
├── docs/                  # documentation website (MkDocs)
├── environment.yml        # conda environment with the paper's package versions
└── pyproject.toml
```

The code that reproduces the paper's analyses and figures is in
[daklab/SpliceVI-utils](https://github.com/daklab/SpliceVI-utils). The `splicevi-extended` branch of this repository keeps
the code as it was during development, with a single-modality splicing model and the evaluation script used then.

---

## Citation

If you use SpliceVI, please cite:

> Vaidyanathan S\*, Isaev K\*, Zweig A, Knowles DA. *Robust Integration of Sparse Single-Cell Alternative Splicing and
> Gene Expression Data with SpliceVI*. bioRxiv 2025.11.26.690853. https://doi.org/10.1101/2025.11.26.690853
>
> \*These authors contributed equally.

```bibtex
@article{splicevi,
  title   = {Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with {SpliceVI}},
  author  = {Vaidyanathan, Smriti and Isaev, Keren and Zweig, Aaron and Knowles, David A},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.11.26.690853}
}
```
