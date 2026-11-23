# Differential Expression & Splicing

SpliceVI supports both **differential gene expression (DE)** and **differential splicing (DS)** analysis between groups of cells. Both use the same underlying Bayesian framework from scvi-tools and differ only in what quantity is being compared and how the effect size is defined. The API mirrors [MultiVI](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/multivi.html), with `differential_splicing` and `get_normalized_splicing` replacing the chromatin accessibility equivalents.

For full background on the statistical framework, see the [scvi-tools differential expression guide](https://docs.scvi-tools.org/en/1.3.3/user_guide/background/differential_expression.html).

---

## How it works

Both DE and DS use `_de_core` from scvi-tools internally, in three steps.

1. **Sample normalized values** from the posterior for each cell in group 1 and group 2 separately, by passing cells through the encoder and then the relevant decoder (expression or splicing)
2. **Compute a per-feature effect size** from those posterior samples
3. **Estimate the posterior probability** that the effect size is at least $\delta$ in one direction (the `"change"` mode)

Each feature is scored by the larger of the two one-sided probabilities, and its log Bayes factor follows from it,

$$
p = \max\{P(\Delta \geq \delta \mid \text{data}),\ P(\Delta \leq -\delta \mid \text{data})\}, \qquad
\text{BF} = \log \frac{p}{1 - p}
$$

Features are called at a posterior expected false discovery rate (default 5%). The calls are the largest set of
features, taken in order of decreasing $p$, whose mean $1 - p$ is at most that rate.

---

## Differential Expression

DE compares **normalized gene expression** between two groups. By default (`norm_expression_function="zinb_expected"`) this is the ZINB expected expression, (1 − dropout probability) × the decoder output (`get_normalized_expression(expected=True)`, see [Imputed Splicing & Expression](imputed.md)). With a ZINB likelihood the dropout probability can absorb part of a difference between groups, which the decoder output alone then misses. `norm_expression_function="decoder"` compares the decoder output, as scVI and MultiVI do, and `"zinb_posterior_mean"` and `"zinb_posterior_sample"` use each cell's ZINB posterior given its observed counts.

The **effect size** follows the standard scVI convention, a log-fold change,

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
    norm_expression_function="zinb_expected",   # the default
)
```

---

## Differential Splicing

DS compares **junction usage (PSI)** between two groups. Because PSI is already on a $[0, 1]$ probability scale, the effect size is a **direct difference** rather than a log-fold change, as for ATAC-seq accessibility scores in scvi-tools,

$$
\text{effect size} = \hat{\psi}^{(2)}_j - \hat{\psi}^{(1)}_j
$$

where $\hat{\psi}^{(k)}_j$ is the PSI for junction $j$ in group $k$. By default (`norm_splicing_function="dm_posterior_sample"`) each Monte Carlo sample draws every cell's PSI from its [DM posterior](imputed.md#dm-normalized-splicing-posterior-mean-psi), so per-cell read uncertainty enters `proba_ds`. `norm_splicing_function="dm_posterior_mean"` uses the posterior mean instead, and `norm_splicing_function="decoder"` the decoder output alone.

```python
ds_results = model.differential_splicing(
    adata=mdata,
    groupby="cell_type",
    group1="Neuron",
    group2="Astrocyte",
    delta=0.10,
    fdr_target=0.05,
    norm_splicing_function="dm_posterior_sample",   # the default
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

- The paper keeps a differential splicing call only when its ATSE has reads in at least three cells (or metacells) of each group. Apply this with `splicevi.filter_read_support(ds_results, min_units=3)`, which does not change the FDR calls themselves
- Both methods support `batch_correction=True` to marginalize over batch effects when comparing groups
- `proba_ds` / `proba_de` are Monte Carlo estimates. Pass `seed=` (for example `seed=0`) to get the same table on repeated calls
- Features with equal posterior probability keep their order in the data, both in the row order of the DE table and in which of them are called when they are tied at the FDR cutoff, so the calls are the same on every machine
