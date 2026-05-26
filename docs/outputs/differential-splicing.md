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

DE compares **normalized gene expression** between two groups. The model function is `get_normalized_expression`, which returns scaled expression values (analogous to scVI's normalized expression).

The **effect size** follows the standard scVI convention — a log-fold change:

$$
\text{LFC} = \log_2(\hat{x}_2 + \epsilon) - \log_2(\hat{x}_1 + \epsilon)
$$

where $\hat{x}_g$ is the posterior mean normalized expression for gene $g$ in group $g$.

### Usage

```python
de_results = model.differential_expression(
    adata=mdata,
    groupby="cell_type",
    group1="Neuron",
    group2="Astrocyte",
    delta=0.25,        # minimum LFC to consider "DE"
    fdr_target=0.05,
)
```

---

## Differential Splicing

DS compares **junction usage (PSI)** between two groups. Because PSI is already on a $[0, 1]$ probability scale, the effect size is a **direct difference** rather than a log-fold change — analogous to how ATAC-seq accessibility scores are handled in scvi-tools:

$$
\text{effect size} = \hat{\psi}^{(2)}_j - \hat{\psi}^{(1)}_j
$$

where $\hat{\psi}^{(k)}_j$ is the posterior mean PSI for junction $j$ in group $k$.

### Normalized PSI: DM posterior mean

By default (`norm_splicing_function="dm_posterior_mean"`), the model uses a **Dirichlet-Multinomial posterior mean** estimate of PSI rather than the raw decoder output. This smooths the decoder's predicted PSI toward the observed data using the learned concentration parameter $c$:

$$
\psi^*_j = \frac{c \cdot p_j + y_j}{c + n_j}
$$

where:

- $p_j$ — decoder-predicted PSI for junction $j$
- $y_j$ — observed junction read count for that cell
- $n_j$ — observed ATSE total read count for that junction's event
- $c$ — learned concentration (either a scalar or per-ATSE, controlled by `dm_concentration`)

When `dm_concentration="atse"`, $c$ is a per-ATSE value mapped to per-junction via the ATSE membership matrix. This gives a data-adaptive shrinkage: cells with high read coverage are pulled toward their observations, while cells with low coverage rely more on the decoder's prediction.

Alternatively, you can use the raw decoder output with `norm_splicing_function="decoder"`.

### Usage

```python
ds_results = model.differential_splicing(
    adata=mdata,
    groupby="cell_type",
    group1="Neuron",
    group2="Astrocyte",
    delta=0.10,         # minimum ΔPSI to consider "DS"
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

- Junctions unobserved in all cells of a group will have `emp_prob = -1` (sentinel for missing)
- Very sparse junctions (low `n_obs_group1/2`) will have wider posteriors — consider filtering before interpretation
- Both methods support `batch_correction=True` to marginalize over batch effects when comparing groups
