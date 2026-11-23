# Latent Space

After training, SpliceVI exposes three latent representations.

| Method | Description |
|--------|-------------|
| `get_latent_representation()` | Joint latent $z$, mixed from both modalities |
| `get_latent_representation(modality="expression")` | Expression-only posterior mean $\mu_e$ |
| `get_latent_representation(modality="splicing")` | Splicing-only posterior mean $\mu_s$ |

---

## Usage

```python
# Joint latent (default)
z = model.get_latent_representation(adata=mdata)   # (cells × n_latent)

# Modality-specific
z_expr = model.get_latent_representation(adata=mdata, modality="expression")
z_spl  = model.get_latent_representation(adata=mdata, modality="splicing")
```

The joint latent can be used with standard tools such as [scanpy](https://scanpy.readthedocs.io/), for example to draw a UMAP.

```python
import scanpy as sc

rna = mdata["rna"]                      # scanpy works on AnnData objects
rna.obsm["X_splicevi"] = z
sc.pp.neighbors(rna, use_rep="X_splicevi")
sc.tl.umap(rna)
sc.pl.umap(rna, color="cell_type")
```

---

## Imputed PSI values

The model also gives imputed junction usage (PSI) for every cell and junction.

```python
psi = model.get_normalized_splicing_DM()   # (cells × junctions), posterior mean given the cell's reads
psi_decoder = model.get_normalized_splicing()   # the decoder's prediction from the latent alone
```

`get_normalized_splicing_DM` combines the decoder's prediction with the cell's own reads, and is the estimate
differential splicing uses by default. See [Imputed Splicing & Expression](imputed.md) for both.
