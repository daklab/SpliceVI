# Imputed Splicing & Expression

A trained model can be applied to other cells with the same genes and junctions as the training data, such as held-out test cells, without retraining. The model's learned parameters stay fixed. For data from a different protocol, the paper first fine-tunes the model's decoders on the new data (`finetune_decoders`).

---

## Normalized expression

`get_normalized_expression` returns the decoder-predicted normalized expression for each gene, analogous to scVI's $\rho_n$. This is the per-gene fraction of total expression predicted by the model, averaged over posterior samples if requested.

```python
expr = model.get_normalized_expression(
    adata=mdata,          # can be held-out data not seen during training
    n_samples=25,
    return_mean=True,
)
# returns DataFrame of shape (cells × genes)
```

With the default zero-inflated negative binomial likelihood, `expected=True` returns the expected expression instead,
$(1 - \pi_{ng})\,\rho_{ng}$ with $\pi_{ng}$ the decoder's dropout probability. Differential expression compares this
quantity by default (`norm_expression_function="zinb_expected"`).

---

## Normalized splicing (decoder output)

`get_normalized_splicing` returns the raw decoder-predicted PSI ($p_{nj}$) for each junction, the model's prediction based solely on the latent representation, before any smoothing toward observed counts.

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

`get_normalized_splicing_DM` returns a **Dirichlet-Multinomial posterior mean** estimate of PSI that blends the decoder prediction with the cell's own observed junction counts,

$$
\psi^*_j = \frac{c \cdot p_j + y_j}{c + n_j}
$$

where

- $p_j$ is the decoder-predicted PSI for junction $j$
- $y_j$ is the observed junction read count for that cell
- $n_j$ is the observed total read count of that junction's ATSE
- $c$ is the learned concentration of the junction's ATSE (one per ATSE by default, or one shared value with `dm_concentration="scalar"`)

This gives **data-adaptive shrinkage**. Cells with high read coverage are pulled toward their own observations, and cells with low coverage rely more on the decoder's prediction. When `dm_concentration="atse"`, $c$ is a per-ATSE value, so the shrinkage differs between ATSEs.

```python
psi_dm = model.get_normalized_splicing_DM(
    adata=mdata,
    n_samples=25,
    return_mean=True,
)
# returns DataFrame of shape (cells × junctions)
```

`differential_splicing` uses the same posterior. By default it draws one sample from it per Monte Carlo sample (`norm_splicing_function="dm_posterior_sample"`), and `norm_splicing_function="dm_posterior_mean"` uses this mean (see [Differential Expression & Splicing](differential-splicing.md)).
