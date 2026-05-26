# Latent Space

After training, SpliceVI exposes three latent representations:

| Method | Description |
|--------|-------------|
| `get_latent_representation()` | Joint latent $z$ — mixed from both modalities |
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

The joint latent can be used directly for downstream analysis (UMAP, clustering, trajectory inference) with standard tools like [scanpy](https://scanpy.readthedocs.io/).

```python
import scanpy as sc

mdata.obsm["X_splicevi"] = z
sc.pp.neighbors(mdata, use_rep="X_splicevi")
sc.tl.umap(mdata)
sc.pl.umap(mdata, color="cell_type")
```

---

## Imputed PSI values

To obtain imputed junction usage probabilities (PSI) from the splicing decoder:

```python
psi = model.get_normalized_splicing()   # (cells × junctions)
```

These are the posterior mean predictions $\mathbb{E}[p_j | z]$ and can be used for visualization or as input to downstream differential splicing analyses.
