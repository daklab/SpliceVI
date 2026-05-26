# Imputed Splicing & Expression

Because SpliceVI is a generative model, its encoder and decoder can be applied to any MuData object that shares the same genes and junctions as the training data. This means you can pass in held-out test data — or even a completely new dataset registered via `setup_mudata` — and extract imputed values without retraining. The model's learned parameters are frozen; the new data simply flows through.

---

## Normalized expression

`get_normalized_expression` returns the decoder-predicted normalized expression for each gene — analogous to scVI's $\rho_n$. This is the per-gene fraction of total expression predicted by the model, averaged over posterior samples if requested.

```python
expr = model.get_normalized_expression(
    adata=mdata,          # can be held-out data not seen during training
    n_samples=25,
    return_mean=True,
)
# returns DataFrame of shape (cells × genes)
```

---

## Normalized splicing (decoder output)

`get_normalized_splicing` returns the raw decoder-predicted PSI ($p_{nj}$) for each junction — the model's prediction based solely on the latent representation, before any smoothing toward observed counts.

```python
psi = model.get_normalized_splicing(
    adata=mdata,
    n_samples=25,
    return_mean=True,
)
# returns DataFrame of shape (cells × junctions)
```

---

## DM-normalized splicing (posterior mean PSI)

`get_normalized_splicing_DM` returns a **Dirichlet-Multinomial posterior mean** estimate of PSI that blends the decoder prediction with the cell's own observed junction counts:

$$
\psi^*_j = \frac{c \cdot p_j + y_j}{c + n_j}
$$

where:

- $p_j$ — decoder-predicted PSI for junction $j$
- $y_j$ — observed junction read count for that cell
- $n_j$ — observed ATSE total read count for that junction's event
- $c$ — learned concentration (scalar or per-ATSE, controlled by `dm_concentration`)

This gives **data-adaptive shrinkage**: cells with high read coverage are pulled toward their own observations; cells with low coverage rely more on the decoder's prediction. When `dm_concentration="atse"`, $c$ is a per-ATSE value, giving event-specific shrinkage.

```python
psi_dm = model.get_normalized_splicing_DM(
    adata=mdata,
    n_samples=25,
    return_mean=True,
)
# returns DataFrame of shape (cells × junctions)
```

This DM-normalized PSI is also the default quantity used internally by `differential_splicing` (see [Differential Expression & Splicing](differential-splicing.md)).
