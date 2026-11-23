# SpliceVI

SpliceVI is a multimodal variational autoencoder that jointly models **gene expression** and **alternative splicing** (junction usage / PSI) from single-cell data. It learns a shared low-dimensional latent space that captures variation across both modalities, enabling tasks like imputation, clustering, trajectory inference, and differential splicing analysis.

SpliceVI follows the same overall architecture and inherits the same scvi-tools mixins as [MultiVI](https://docs.scvi-tools.org/en/1.3.3/user_guide/models/multivi.html) (Ashuach et al., 2021), which jointly models RNA-seq and chromatin accessibility. The key difference is that the ATAC branch is replaced throughout by a **splicing-specific branch** — a missingness-aware encoder, a splicing-specific decoder, and splicing likelihoods (Binomial / Beta-Binomial / Dirichlet-Multinomial) suited to the PSI scale. The DE/DS analysis API mirrors MultiVI's, with `get_normalized_splicing` and `differential_splicing` replacing the accessibility equivalents.

> Ashuach T\*, Gabitto MI\*, Jordan MI, Yosef N. *MultiVI: deep generative model for the integration of multi-modal data.* bioRxiv, 2021.

---

## Model overview

SpliceVI uses two separate encoder branches — one for gene expression and one for splicing — whose posteriors are mixed into a joint latent space. Two decoder branches reconstruct each modality from the joint latent.

```
Gene expression ──► GE Encoder ──► q(z|x_e) ──┐
                                                ├──► Mix ──► z ──► GE Decoder ──► x_e
Splicing (PSI) ───► Splice Encoder ──► q(z|x_s) ─┘              └──► Splice Decoder ──► x_s
```

### Splicing encoder

Because splicing observations are highly sparse — many junctions are unobserved in a given cell — SpliceVI uses a **missingness-aware partial encoder** (`PartialEncoderEDDI`). The design draws inspiration from the [EDDI framework](https://arxiv.org/abs/1809.11142) (Ma et al., 2019), which handles pervasive missing values in VAEs using feature embeddings and exchangeable neural networks (see also the [SpliceVI preprint](https://www.biorxiv.org/content/10.1101/2025.11.26.690853v1)). The encoder:

1. Embeds each *observed* junction independently via a shared $h$ network
2. Pools the per-junction embeddings (mean or sum) across all observed junctions in a cell
3. Maps the pooled representation through an MLP to produce posterior statistics $(\mu, \sigma^2)$

This avoids any imputation of missing values at the input level and is more principled than masking or zero-filling.

### Splicing likelihood

Three options are supported:

| Option | Description |
|--------|-------------|
| `binomial` | Simple binomial over junction read counts |
| `beta_binomial` | Beta-binomial with a learned per-junction concentration $\phi_j$ |
| `dirichlet_multinomial` | Dirichlet-multinomial grouped per ATSE event; enforces that junction probabilities within an event sum to 1 |

The default and recommended option is `dirichlet_multinomial`.

### Modality mixing

The two per-modality posteriors are combined into a single joint posterior before sampling. The mixing strategy is controlled by the `modality_weights` argument, and follows the same parameterization as MultiVI:

| Option | Description |
|--------|-------------|
| `equal` | Fixed 50/50 average |
| `universal` | One global learned weight per modality |
| `cell` | Per-cell learned weights |
| `concatenate` | Concatenate both latents — no mixing, doubled latent size |

---

## Architecture

| Component | Default | Options |
|-----------|---------|---------|
| Splicing encoder | `partial` | `vanilla`, `partial` |
| Splicing decoder | `vanilla` | `vanilla`, `linear` |
| Expression decoder | `vanilla` | `vanilla`, `linear` |
| Splicing likelihood | `dirichlet_multinomial` | `binomial`, `beta_binomial`, `dirichlet_multinomial` |
| Latent distribution | `normal` | `normal`, `ln` |

---

## Key hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_latent` | 30 | Joint latent dimensionality |
| `n_hidden` | auto | Width of encoder/decoder MLPs |
| `n_layers_encoder` | 2 | Depth of encoder networks |
| `dropout_rate` | 0.01 | Dropout in MLPs |
| `code_dim` | 64 | Per-junction embedding size (partial encoder) |
| `h_hidden_dim` | 128 | Hidden width of per-junction $h$ network |
| `encoder_hidden_dim` | 128 | Hidden width of the post-pooling MLP |
| `psi_input` | `deviation` | Partial-encoder input: junction usage minus its population usage (`raw` = usage itself) |
| `decoder_intercept_init` | `population` | Splicing-decoder bias starts at the population junction usage |
| `decoder_depth_covariates` | `True` | Depth inputs (detected genes, observed junctions, library size) into both decoders |
| `modality_penalty` | `None` | Alignment penalty between the two posteriors (`Jeffreys`, `MMD`) |
| `variance_mixing` | `squared` | How the two posterior variances combine into the joint variance |

These defaults are the settings of the paper's models; `SPLICEVI.preset("paper")` returns the same values, including
the training settings.

---

## API reference

::: splicevi.SPLICEVI
