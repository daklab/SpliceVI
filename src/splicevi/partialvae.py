from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from typing import TYPE_CHECKING, Literal, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli, kl_divergence
from torch.nn.functional import one_hot
import torch.nn.functional as F

from scvi import REGISTRY_KEYS, settings
from scvi.module._constants import MODULE_KEYS
from scvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from scvi.nn import FCLayers


if TYPE_CHECKING:
    from collections.abc import Callable
    from torch.distributions import Distribution

logger = logging.getLogger(__name__)

from torch.special import gammaln

def group_logsumexp(A, B):
    """
    Computes the log-sum-exp of B, grouped by A.
    A is a sparse matrix (must be coalesced already!) with shape (P, G), where P is the number of variables (junctions)
    and G is the number of groups (ATSEs). B is a dense matrix with shape (N, P), where
    N is the number of samples (cells). The result is a dense matrix with shape (N, G).
    """
    idx_p, idx_g = A.indices()              # (2, P), since each var belongs to one group
    idx_p = idx_p.to(B.device)
    idx_g = idx_g.to(B.device)

    N, P = B.shape
    G = A.shape[1]

    # (N, P) → gather relevant group index for each variable
    group_idx = torch.empty(P, dtype=torch.long, device=B.device)
    group_idx[idx_p] = idx_g

    # Get max within each group
    max_B = torch.full((N, G), float('-inf'), device=B.device)
    max_B.scatter_reduce_(1, group_idx.expand(N, -1), B, reduce='amax')

    # Center B by group max
    B_centered = B - max_B.gather(1, group_idx.expand(N, -1))

    exp_B_centered = torch.exp(B_centered)

    # Sum exp over each group
    sum_exp = torch.zeros((N, G), device=B.device)
    sum_exp.scatter_add_(1, group_idx.expand(N, -1), exp_B_centered)

    return torch.log(sum_exp + 1e-8) + max_B

def subtract_group_logsumexp(A, B, group_logsumexp):
    """
    Subtracts the group log-sum-exp from each element in B, grouped by A.
    A is a sparse matrix (must be coalesced already!) with shape (P, G), where P is the number of variables (junctions)
    and G is the number of groups (ATSEs). B is a dense matrix with shape (N, P), where
    N is the number of samples (cells). The result is a dense matrix with shape (N, P).
    """
    idx_p, idx_g = A.indices()
    idx_g = idx_g.to(B.device)
    N = B.size(0)
    return B - group_logsumexp[:, idx_g]

def nbetaln(count, alpha):  
    """
    Computes log(count * beta(count, alpha)) == gammaln(count + 1) + gammaln(alpha) - gammaln(count + alpha). 
    Note this is zero when count == 0, which could potentially be used to improve efficiency.
    """
    return gammaln(alpha) + gammaln(count+1.) - gammaln(alpha + count)

def dirichlet_multinomial_likelihood(
    counts: torch.Tensor,        # (N, P)
    atse_counts: torch.Tensor,   # (N, G)
    junc2atse: torch.sparse.Tensor,  # (P, G)
    alpha: torch.Tensor,         # (N, P)
    mask: torch.Tensor | None = None,  # optional (N, P)
) -> torch.Tensor:
    """
    Computes the batch-averaged Dirichlet–multinomial log-likelihood,
    **masking out** any junctions where mask==0 and any ATSEs where atse_counts==0.

    Returns:
        scalar = mean_over_cells( sum_per_ATSE(LL_atse * ATSE_mask)
                                - sum_per_junction(LL_junc * junc_mask) )
    """
    device = counts.device
    atse_counts = atse_counts.to(device)
    alpha       = alpha.to(device)
    mask        = mask.to(device) if mask is not None else None
    junc2atse   = junc2atse.to(device)
    N, P = counts.shape
    G     = junc2atse.shape[1]

    idx_p, idx_g = junc2atse.indices()
    idx_p = idx_p.to(counts.device)
    idx_g = idx_g.to(counts.device)
    idx_p, idx_g = junc2atse.indices()
    idx_p = idx_p.to(device)
    idx_g = idx_g.to(device)             # each length ≈ P

    true_atse = torch.zeros((N, G), device=device, dtype=atse_counts.dtype)
    true_atse.scatter_reduce_(
        dim=1,
        index=idx_g.expand(N, -1),                    # (N, P)
        src=atse_counts,                              # (N, P)
        reduce="amax",
    )
    atse_counts = true_atse                           # (N, G)

    # 2) compute group-sums of alpha → (N, G)
    alpha_sums = alpha @ junc2atse                    # (N, G)

    # 3) per-ATSE log-likelihoods
    ll_atse = nbetaln(atse_counts, alpha_sums)        # (N, G)
    # mask out any ATSEs with zero count
    atse_mask = (atse_counts > 0).to(ll_atse.dtype)   # (N, G)
    ll_atse = ll_atse * atse_mask                     # zeros where atse_count==0
    # sum per ATSE and then average over cells
    per_atse = ll_atse.sum(dim=1).mean()              # scalar

    # 4) per-junction log-likelihoods
    ll_junc = nbetaln(counts, alpha)                  # (N, P)
    if mask is not None:
        mask_bool = mask.to(torch.bool)               # (N, P)
        ll_junc = ll_junc * mask_bool.to(ll_junc.dtype)
    # sum per junction and then average over cells
    per_junc = ll_junc.sum(dim=1).mean()              # scalar

    # 5) return batch-averaged difference
    return per_atse - per_junc



def binomial_loss_function(
    logits: torch.Tensor,
    junction_counts: torch.Tensor,
    cluster_counts: torch.Tensor,
    n,
    k,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Reconstruction Binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      mask: Mask to exclude missing values when running partialVAE.
      
    Returns:
      total_loss: Reconstruction loss according to binomial likelihood.
    """

    if mask is not None:
        mask = mask.to(torch.bool)  # ← convert floats/ints to bool
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        cluster_counts = cluster_counts[mask]

    probs = torch.sigmoid(logits)
    log_p = torch.log(probs + 1e-8)
    log_1_p = torch.log(1 - probs + 1e-8)
    log_lik = (
        junction_counts * log_p + (cluster_counts - junction_counts) * log_1_p
    )
    return -log_lik.mean()


def beta_binomial_log_pmf(k, n, alpha, beta):
    """
    Calculate the log probability mass function for the beta-binomial distribution.
    
    Parameters:
      k: Number of successes.
      n: Number of trials.
      alpha: Alpha parameter of beta distribution.
      beta: Beta parameter of beta distribution.
      
    Returns:
      log_pmf: Log of probability mass function.
    """
    return (
        torch.lgamma(n + 1)
        - torch.lgamma(k + 1)
        - torch.lgamma(n - k + 1)
        + torch.lgamma(k + alpha)
        + torch.lgamma(n - k + beta)
        - torch.lgamma(alpha)
        - torch.lgamma(beta)
        + torch.lgamma(alpha + beta)
        - torch.lgamma(n + alpha + beta)
    )


def beta_binomial_loss_function(
    logits: torch.Tensor,
    junction_counts: torch.Tensor,
    cluster_counts: torch.Tensor,
    n: int,
    k: int,
    concentration: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Beta-binomial loss function for VAE.
    
    Parameters:
      logits: Reconstructed logits from decoder.
      junction_counts: Junction counts from data.
      cluster_counts: Cluster counts from data.
      n: Number of samples in dataset.
      k: Number of batches in dataloader.
      concentration: Concentration parameter.
      mask: Optional mask to exclude missing values.
      
    Returns:
      total_loss: Combined reconstruction and KL loss.
    """
    if mask is not None:
        mask = mask.to(torch.bool)  # ← convert floats/ints to bool
        logits = logits[mask]
        junction_counts = junction_counts[mask]
        cluster_counts = cluster_counts[mask]


    probs = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    alpha = probs * concentration
    beta = (1.0 - probs) * concentration
    log_lik = beta_binomial_log_pmf(
        k=junction_counts, n=cluster_counts, alpha=alpha, beta=beta
    )
    log_lik = log_lik #* (float(n) / float(k))
    return -log_lik.mean()


import torch
from torch import nn
from collections.abc import Iterable
from scvi.nn import FCLayers


class PartialEncoderEDDIFaster(nn.Module):
    """
    Fast EDDI partial encoder, batched.

    Shapes / notation:
      B = batch size (cells)
      J = number of junctions
      D = code_dim
      Z = latent_dim

    """
    def __init__(
        self,
        input_dim: int,
        code_dim: int,
        h_hidden_dim: int,
        encoder_hidden_dim: int,
        latent_dim: int,
        dropout_rate: float = 0.0,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
        inject_covariates: bool = True,
        pool_mode: Literal["mean", "sum"] = "mean",
        max_nobs: int = -1,
        encoder_n_layers: int = 2,    
        psi_input: Literal["raw", "centred", "deviation"] = "raw",   # "centred": psi - population psi; "deviation": centred input, h(0) subtracted, fixed pool divisor
    ):
        super().__init__()
        ...

        super().__init__()
        self.code_dim = code_dim
        self.pool_mode = pool_mode
        self.psi_input = psi_input
        if psi_input in ("centred", "deviation"):
            self.register_buffer("pop_psi", torch.zeros(input_dim))   # only registered when used, so older checkpoints still load
        if psi_input == "deviation":   # pooled = sum_j (h_j - h(0)) / pool_divisor (training median of observed junctions per cell)
            if pool_mode != "mean":
                raise ValueError("psi_input='deviation' replaces the pooling divisor; use pool_mode='mean'")
            self.register_buffer("pool_divisor", torch.ones(()))
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont = n_cont
        self.inject_covariates = inject_covariates
        self.max_nobs = max_nobs

        # Per-junction embedding table (J, D)
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim))

        # h-layer: [psi (1) | F_j (D)] -> D
        in_dim_h = 1 + code_dim

        

        self.h_layer = nn.Sequential(
            nn.Linear(in_dim_h, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(h_hidden_dim, code_dim),
            nn.LayerNorm(code_dim),
            nn.ReLU(),
        )

        # Cell-level encoder MLP → (mu, logvar)
        self.encoder_mlp = FCLayers(
            n_in=code_dim,
            n_out=2 * latent_dim,
            n_cat_list=n_cat_list or [],
            n_cont=n_cont,
            n_layers=encoder_n_layers,
            n_hidden=encoder_hidden_dim,
            dropout_rate=dropout_rate,
            use_batch_norm=False,
            use_layer_norm=True,
            inject_covariates=inject_covariates,
        )


    def _h(self, h_in: torch.Tensor) -> torch.Tensor:
        """Per-junction network. With psi_input="deviation" the output at zero deviation, h(0), is subtracted (dropout skipped in
        both passes, so an observed junction at its population ratio contributes exactly 0 in training and evaluation)."""
        if self.psi_input != "deviation":
            return self.h_layer(h_in)
        def run(v):
            for m in self.h_layer:
                if not isinstance(m, nn.Dropout):
                    v = m(v)
            return v
        out = run(h_in) - run(torch.zeros(1, h_in.shape[1], device=h_in.device, dtype=h_in.dtype))
        # exact zero where the deviation is exactly zero (the two passes can differ at float rounding, ~1e-7, across GEMM batch sizes)
        return out * (h_in[:, :1] != 0).to(out.dtype)

    def forward(
    self,
    x: torch.Tensor,                 # (B, J)
    mask: torch.Tensor,              # (B, J), 1=observed, 0=missing
    *cat_list: torch.Tensor,
    cont: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Observed-only path:
        - Gather (b, j) pairs where mask==1
        - Run h_layer only on those rows
        - Sum (or mean) per cell
        - Encode to (mu, logvar)
        """
        B, J = x.shape
        D = self.code_dim
        device = x.device
        dtype  = x.dtype

        # Ensure boolean mask (no grad)
        mask_bool = mask.bool()

        # Find observed indices (vectors of shape (N_obs,))
        b_idx, j_idx = mask_bool.nonzero(as_tuple=True)
        N_obs = b_idx.numel()
        F_j = self.feature_embedding

        # Early-exit if nothing is observed
        if N_obs == 0:
            pooled = torch.zeros(B, D, device=device, dtype=dtype)
            mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)  # (B, 2Z)
            mu, logvar = mu_logvar.chunk(2, dim=-1)
            return mu, logvar

        # Precompute a normalized copy once (avoids re-normalizing the same rows repeatedly)
        # Do this where F_j is defined / available before the if/else:
        F_j_norm = F.normalize(F_j, p=2, dim=1, eps=1e-8)   # (J, D)

        if (self.max_nobs < 0) or (N_obs <= self.max_nobs):
            # ---- Original (no chunking) path ----
            x_obs = x[b_idx, j_idx].unsqueeze(1)                  # (N_obs, 1)
            if self.psi_input in ("centred", "deviation"):
                x_obs = x_obs - self.pop_psi[j_idx].unsqueeze(1)     # deviation from population psi
            F_obs = F_j_norm.index_select(0, j_idx)               # (N_obs, D), already L2-normalized

            F_obs_scaled = F_obs * x_obs                          # broadcast scale by usage ratio
            h_in  = torch.cat([x_obs, F_obs_scaled], dim=1)       # (N_obs, 1 + D)
            h_obs = self._h(h_in)                                 # (N_obs, D)

            pooled = torch.zeros(B, D, device=device, dtype=h_obs.dtype)
            pooled.index_add_(0, b_idx, h_obs)

        else:
            # ---- Chunked path: same result, steadier memory ----
            pooled = None
            for start in range(0, N_obs, self.max_nobs):
                end = min(start + self.max_nobs, N_obs)

                bi = b_idx[start:end]                             # (n,)
                jj = j_idx[start:end]                             # (n,)

                x_chunk = x[bi, jj].unsqueeze(1)                  # (n, 1)
                if self.psi_input in ("centred", "deviation"):
                    x_chunk = x_chunk - self.pop_psi[jj].unsqueeze(1)
                F_chunk = F_j_norm.index_select(0, jj)            # (n, D), already L2-normalized

                F_chunk_scaled = F_chunk * x_chunk                # scale by usage ratio
                h_in  = torch.cat([x_chunk, F_chunk_scaled], dim=1)  # (n, 1 + D)
                h_out = self._h(h_in)                                # (n, D)

                if pooled is None:
                    pooled = torch.zeros(B, D, device=device, dtype=h_out.dtype)
                pooled.index_add_(0, bi, h_out)


        # Mean pooling if requested
        if self.psi_input == "deviation":   # fixed divisor, so the pool does not carry the event count
            pooled = pooled / self.pool_divisor
        elif self.pool_mode == "mean":
            # counts per cell (B,), keep at least 1 to avoid division by zero
            counts = torch.bincount(b_idx, minlength=B).to(pooled.dtype).view(B, 1).clamp_min_(1)
            pooled = pooled / counts

        # Final per-cell projection → (mu, logvar)
        mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)  # (B, 2Z)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar


class LinearDecoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        n_cat_list: Iterable[int] | None = None,
        n_cont: int = 0,
    ):
        super().__init__()
        # track covariates exactly as in encoder
        self.n_cat_list = [n for n in (n_cat_list or []) if n > 1]
        self.n_cont     = n_cont
        total_cov       = sum(self.n_cat_list) + self.n_cont

        # single linear layer: input = [z, covariates]
        self.linear = nn.Linear(latent_dim + total_cov, output_dim)

    def forward(
        self,
        z: torch.Tensor,              # (B, latent_dim)
        *cat_list: torch.Tensor,      # each (B,1)
        cont: torch.Tensor | None = None,  # (B, n_cont)
    ) -> torch.Tensor:
        
        covs: list[torch.Tensor] = []
        if cont is not None:
            covs.append(cont)
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            # one-hot encode each categorical and append
            oh = F.one_hot(cat.squeeze(-1), n_cat).float()
            covs.append(oh)
        # final input to linear: (B, latent_dim + total_cov)
        inp = torch.cat([z, *covs], dim=1) if covs else z
        return self.linear(inp)

class PARTIALVAE(BaseModuleClass):
    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,  # (unused; you can delete if you really don't care)
        n_latent: int = 10,
        n_continuous_cov: int = 0,
        n_cats_per_cov: list[int] | None = None,
        dropout_rate: float = 0.1,
        splice_likelihood: Literal["binomial", "beta_binomial", "dirichlet_multinomial"] = "dirichlet_multinomial",
        latent_distribution: Literal["normal", "ln"] = "normal",
        encode_covariates: bool = False,
        deeply_inject_covariates: bool = True,
        learn_concentration: bool = True,
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        encoder_hidden_dim: int = 128,
        encoder_n_layers: int = 2,          # NEW: #layers in encoder FCLayers
        pool_mode: Literal["mean", "sum"] = "mean",
        max_nobs: int = -1,
    ):
        super().__init__()
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates

        self.splice_likelihood = splice_likelihood
        self.latent_distribution = latent_distribution

        self.learn_concentration = learn_concentration
        self.code_dim = code_dim
        self.pool_mode = pool_mode
        self.max_nobs = max_nobs

        self.extra_payload_autotune = False  # if you don't use this anywhere, drop everywhere

        # AnnData dimensions
        self.input_dim = n_input
        self.n_continuous_cov = n_continuous_cov
        self.n_cats_per_cov = n_cats_per_cov or []

        cat_list = [n_batch] + list(n_cats_per_cov) if n_cats_per_cov is not None else []
        encoder_cat_list = cat_list if encode_covariates else None

        # shared concentration parameter ϕ
        if learn_concentration:
            self.log_concentration = nn.Parameter(torch.tensor(4.6))
        else:
            self.log_concentration = None

        print(f"Using code_dim={code_dim}!")
        print(f"Using latent_dim={n_latent}")
        print(f"Encoder pool_mode: {pool_mode}, max_nobs={max_nobs}")

        # --- ONLY encoder: PartialEncoderEDDIFaster ---
        self.encoder = PartialEncoderEDDIFaster(
            input_dim=n_input,
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            latent_dim=n_latent,
            dropout_rate=dropout_rate,
            n_cat_list=encoder_cat_list,
            n_cont=n_continuous_cov,
            inject_covariates=encode_covariates,
            pool_mode=pool_mode,
            max_nobs=max_nobs,
            encoder_n_layers=encoder_n_layers,
        )

        if latent_distribution == "ln":
            self.encoder.z_transformation = nn.Softmax(dim=-1)
        else:
            self.encoder.z_transformation = lambda x: x

        self.decoder = LinearDecoder(
            latent_dim=n_latent,
            output_dim=n_input,
            n_cat_list=cat_list,
            n_cont=n_continuous_cov,
        )

        # store for later (optional)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_n_layers = encoder_n_layers


    @auto_move_data
    def _get_inference_input(self, tensors):
        return {
            MODULE_KEYS.X_KEY: tensors[REGISTRY_KEYS.X_KEY],        # your junction ratios
            "mask":           tensors.get(REGISTRY_KEYS.PSI_MASK_KEY),
            "batch_index":    tensors[REGISTRY_KEYS.BATCH_KEY],     # integer batch code
            "cat_covs":       tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),  # stacked one‐hot columns
            "cont_covs":      tensors.get(REGISTRY_KEYS.CONT_COVS_KEY), # numeric covariates
        }

    @auto_move_data
    def inference(
        self,
        x,
        mask,
        batch_index,
        cat_covs=None,
        cont_covs=None,
        n_samples=1,
    ) -> dict[str, torch.Tensor]:
        # 1) prepare encoder inputs (concatenate continuous covariates if requested)
        if cont_covs is not None and self.encode_covariates:
            x_input = torch.cat([x, cont_covs], dim=-1)
        else:
            x_input = x

        # 2) split out any one-hot categorical covariates
        if cat_covs is not None and self.encode_covariates:
            cat_list = torch.split(cat_covs, 1, dim=1)
        else:
            cat_list = ()

        # 3) run encoder → returns (mu, raw_logvar)
        #    we still pass batch_index as the first “category”
        mu, raw_logvar = self.encoder(
            x_input, mask, batch_index, *cat_list, cont=cont_covs
        )

        # --- ensure logvar is in a safe range ---
        # prevents under/overflow in exp()
        logvar = torch.clamp(raw_logvar, min=-10.0, max=10.0)

        # 4) build the posterior Normal
        var   = torch.exp(logvar)         # variance = exp(logvar) > 0
        sigma = torch.sqrt(var)           # standard deviation
        qz    = Normal(mu, sigma)

        # 5) sample (rsample so that gradients flow)
        if n_samples == 1:
            z = qz.rsample()
        else:
            z = qz.sample((n_samples,))

        # 6) apply any “z_transformation” (e.g. identity or softmax)
        z = self.encoder.z_transformation(z)

        return {
            "z":    z,      # (B, Z) or (n_samples, B, Z)
            "qz_m": mu,     # posterior means
            "qz_v": var,    # **variance**, not logvar
            "x":    x,
        }


    def _get_generative_input(self, tensors, inference_outputs):
        return {
            MODULE_KEYS.Z_KEY:   inference_outputs["z"],
            "qz_m":              inference_outputs["qz_m"],      # <<< add this
            "batch_index":       tensors[REGISTRY_KEYS.BATCH_KEY],
            "cat_covs":          tensors.get(REGISTRY_KEYS.CAT_COVS_KEY),
            "cont_covs":         tensors.get(REGISTRY_KEYS.CONT_COVS_KEY),
        }

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        cat_covs=None,
        cont_covs=None,
        use_z_mean=False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        # if you want to use the posterior mean:
        if use_z_mean:
            z = qz_m

        # again split cat_covs → tuple
        if cat_covs is not None:
            cat_list = torch.split(cat_covs, 1, dim=1)
        else:
            cat_list = ()

        # prepare decoder input by concatenating cont_covs if desired
        decoder_input = (
            torch.cat([z, cont_covs], dim=-1)
            if (cont_covs is not None and self.deeply_inject_covariates)
            else z
        )

        # finally call your decoder, passing batch_index first:
        reconstruction = self.decoder(decoder_input, batch_index, *cat_list, cont=cont_covs)

        return {"reconstruction": reconstruction}   # now shape (J,)



    def loss(
        self,
        tensors: dict[str, torch.Tensor],
        inference_outputs: dict[str, torch.Tensor],
        generative_outputs: dict[str, torch.Tensor],
        kl_weight: float | torch.Tensor = 1.0,
    ) -> LossOutput:
        # 1) unpack data
        x = tensors[REGISTRY_KEYS.X_KEY]                      # (B, D) input ratios (unused here)
        mask = tensors.get(REGISTRY_KEYS.PSI_MASK_KEY, None)  # (B, D) binary
        junc = tensors["junction_counts"]                     # (B, D) successes
        clus = tensors["cluster_counts"]                      # (B, D) trials

        # 2) reconstruction
        p = generative_outputs["reconstruction"]  # (B, D) Bernoulli probabilities
        if self.splice_likelihood == "binomial":
            reconst = binomial_loss_function(
                logits=p,
                junction_counts=junc,
                cluster_counts=clus,
                n=x.numel(),
                k=x.shape[0],
                mask=mask,
            )

        elif self.splice_likelihood == "beta_binomial":
            concentration = torch.exp(self.log_concentration)
            reconst = beta_binomial_loss_function(
                logits=p,
                junction_counts=junc,
                cluster_counts=clus,
                n=x.numel(),
                k=x.shape[0],
                concentration=concentration,
                mask=mask,
            )

        elif self.splice_likelihood == "dirichlet_multinomial":
            # --- prepare inputs ---
            # junc:        (B,D)  = counts per junction per cell
            # clus:        (B,D)  = repeated ATSE totals per junction per cell
            # self.junc2atse: sparse (D, G)

            # decoder output p is currently a “logit” tensor of shape (B,D)
            logits = p

            # we need atse_counts of shape (B, D), which you already have in `clus`.
            atse_counts = clus

            # concentration scalar fixed at 10.0 for now
            
            concentration = torch.exp(self.log_concentration)

            # --- turn logits into per‐group softmax‐logits ---
            # group_logsumexp and subtract_group_logsumexp expect (N,P)→(N,G)
            lse = group_logsumexp(self.junc2atse, logits)            # → (N,G)
            softmax_logits = subtract_group_logsumexp(
                self.junc2atse, logits, lse
            )                                                           # → (N,P)

            # build alpha = conc * exp(softmax_logits)
            alpha = concentration * torch.exp(softmax_logits)

            # compute the DM log‐likelihood
            ll = dirichlet_multinomial_likelihood(
                counts=junc,
                atse_counts=atse_counts,
                junc2atse=self.junc2atse,
                alpha=alpha,
                mask = mask,
            )
            # negative mean‐log‐lik
            reconst = -ll

        else:
            raise ValueError(f"Unknown splice_likelihood={self.splice_likelihood}")

        # reconst is a scalar: -E_q [ log p(x|z) ]

        # 3) KL divergence
        mu = inference_outputs["qz_m"]   # (B, Z)
        var = inference_outputs["qz_v"]  # (B, Z) = σ²
        sigma = torch.sqrt(var)          # (B, Z)
        qz = Normal(mu, sigma)
        pz = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
        kl_local = kl_divergence(qz, pz).sum(dim=1)  # (B,) sum over latent dims

        # 4) total ELBO
        #    average KL over batch, add to reconst
        total = reconst + kl_weight * kl_local.mean()

        # 5) wrap in LossOutput
        return LossOutput(
            loss=total,
            reconstruction_loss=reconst,
            kl_local=kl_local,
            n_obs_minibatch=x.shape[0],
        )

    @torch.inference_mode()
    def sample(self, tensors: Dict[str, torch.Tensor], n_samples: int = 1) -> torch.Tensor:
        _, gen_out = self.forward(tensors, compute_loss=False)
        return gen_out["reconstruction"].cpu() 