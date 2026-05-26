# Differential Expression & Splicing

SpliceVI supports both **differential gene expression (DE)** and **differential splicing (DS)** analysis between groups of cells. Both use the same underlying Bayesian framework from scvi-tools — they differ only in what quantity is being compared and how the effect size is defined. The API mirrors [MultiVI](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/multivi.html), with `differential_splicing` and `get_normalized_splicing` replacing the chromatin accessibility equivalents.

For full background on the statistical framework, see the [scvi-tools differential expression guide](https://docs.scvi-tools.org/en/1.3.3/user_guide/background/differential_expression.html).

---

## How it works

Both DE and DS use `_de_core` from scvi-tools internally. The key idea is:

1. **Sample normalized values** from the posterior for each cell in group 1 and group 2 separately, by passing cells through the encoder and then the relevant decoder (expression or splicing)
2. **Compute a per-feature effect size** from those posterior samples
3. **Estimate the posterior probability** that the effect size exceeds a threshold $\delta$ (the `"change"` mode)

The per-feature Bayes factor is then:

$$
\text{BF} = \log \frac{P(|\Delta| > \delta \mid \text{data})}{P(|\Delta| \leq \delta \mid \text{data})}
$$

An FDR-controlled call of differential features is made using a target FDR threshold (default 5%).

---

## Differential Expression

DE compares **normalized gene expression** between two groups using `get_normalized_expression` (see [Imputed Splicing & Expression](imputed.md)).

The **effect size** follows the standard scVI convention — a log-fold change:

$$
\text{LFC} = \log_2(\hat{x}^{(2)}_g + \epsilon) - \log_2(\hat{x}^{(1)}_g + \epsilon)
$$

where $\hat{x}^{(k)}_g$ is the posterior mean normalized expression for gene $g$ in group $k$.

```python
de_results = model.differential_expression(
    adata=mdata,
    groupby="cell_type",
    group1="Neuron",
    group2="Astrocyte",
    delta=0.25,
    fdr_target=0.05,
)
```

---

## Differential Splicing

DS compares **junction usage (PSI)** between two groups. Because PSI is already on a $[0, 1]$ probability scale, the effect size is a **direct difference** rather than a log-fold change — analogous to how ATAC-seq accessibility scores are handled in scvi-tools:

$$
\text{effect size} = \hat{\psi}^{(2)}_j - \hat{\psi}^{(1)}_j
$$

where $\hat{\psi}^{(k)}_j$ is the posterior mean PSI for junction $j$ in group $k$, computed using the [DM posterior mean](imputed.md#dm-normalized-splicing-posterior-mean-psi) by default (`norm_splicing_function="dm_posterior_mean"`). You can switch to the raw decoder output with `norm_splicing_function="decoder"`.

```python
ds_results = model.differential_splicing(
    adata=mdata,
    groupby="cell_type",
    group1="Neuron",
    group2="Astrocyte",
    delta=0.10,
    fdr_target=0.05,
    norm_splicing_function="dm_posterior_mean",   # recommended
)
```

### Output columns

| Column | Description |
|--------|-------------|
| `proba_ds` | Posterior probability of differential splicing |
| `is_ds_fdr` | Boolean FDR-controlled call at `fdr_target` |
| `bayes_factor` | Log Bayes factor |
| `effect_size` | $\hat{\psi}^{(2)} - \hat{\psi}^{(1)}$ (model posterior means) |
| `emp_effect` | Empirical $\bar{\psi}^{(2)} - \bar{\psi}^{(1)}$ from observed PSI values |
| `est_prob1` / `est_prob2` | Model posterior mean PSI per group |
| `emp_prob1` / `emp_prob2` | Empirical mean PSI per group (observed cells only) |
| `n_obs_group1` / `n_obs_group2` | Number of cells with observed data per junction per group |

---

## Notes

- Very sparse junctions (low `n_obs_group1/2`) will have wider posteriors — consider filtering before interpretation
- Both methods support `batch_correction=True` to marginalize over batch effects when comparing groups
