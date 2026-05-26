# SpliceVI

Multimodal VAE for joint modeling of alternative splicing (PSI) and gene expression from single-cell data. Built on [scvi-tools](https://github.com/scverse/scvi-tools).

[SpliceVI Docs](https://daklab.github.io/SpliceVI/)

---

## Installation

### 1. Create and activate the environment

```bash
conda create -n splicevi-env python=3.12
conda activate splicevi-env
```

### 2. Install the package

```bash
git clone https://github.com/daklab/SpliceVI.git
cd SpliceVI
pip install -e .
```

This installs all dependencies automatically from `pyproject.toml`. For W&B logging, additionally run `pip install wandb`.

After installation, `from splicevi import SPLICEVI` works from any script or notebook.

> **Exact environment:** `requirements.txt` contains a full freeze of the development environment for reproducibility reference, but has machine-specific paths and is not intended as a direct install source.

---

## Repository structure

```
SpliceVI/
├── src/splicevi/
│   ├── splicevi.py          # SPLICEVI model class for GE+AS (training, inference, DE, DS)
│   ├── splicevae.py         # VAE module for GE+AS (encoder/decoder architecture)
│   ├── partialvae.py        # AS Missingness-Aware Partial VAE module
│   ├── eddisplice.py        # EDDISPLICE model class for single-modality AS-only VAE using PARTIALVAE
│   └── __init__.py          # Package exports
├── train_splicevi.py        # Training entry point
├── eval_splicevi.py         # Evaluation entry point (UMAPs, metrics, imputation)
├── train_splicevi.sh        # SLURM job script for training
├── eval_splicevi.sh         # SLURM job script for evaluation
├── pyproject.toml           # Package configuration
└── requirements.txt         # Full conda environment freeze
```

---

## Usage

Configure paths and hyperparameters at the top of the shell scripts, then submit:

```bash
sbatch train_splicevi.sh
sbatch eval_splicevi.sh
```

Or run directly:

```bash
python train_splicevi.py --help
python eval_splicevi.py --help
```

### Future additions:
- [ ] Add `tutorial.ipynb` for a walkthrough of model setup, training, and application to other datasets.
- [ ] Add trained models to Hugging Face
---

## Citation

If you use SpliceVI in your work, please cite:

> Vaidyanathan S, Isaev K, Zweig A, Knowles DA. *Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with SpliceVI*. bioRxiv 2025.11.26.690853. https://doi.org/10.1101/2025.11.26.690853

```bibtex
@article{splicevi,
  title   = {Robust Integration of Sparse Single-Cell Alternative Splicing and Gene Expression Data with {SpliceVI}},
  author  = {Vaidyanathan, Smriti and Isaev, Keren and Zweig, Aaron and Knowles, David A},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.11.26.690853}
}
```
