from __future__ import annotations

import logging
from contextlib import contextmanager
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from mudata import MuData
from sklearn.decomposition import TruncatedSVD
from torch.distributions import Normal
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.model._utils import (
    _get_batch_code_from_category,
    scrna_raw_counts_properties,
    use_distributed_sampler,
)
from scvi.module.base import (
    BaseModuleClass,
)
from scvi.model.base import (
    ArchesMixin,
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)

from scvi.model.base._de_core import _de_core
from .splicevae import SPLICEVAE
from scvi.train import AdversarialTrainingPlan
from scvi.utils import track
from scvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from typing import Literal

    from anndata import AnnData
    from scvi._types import AnnOrMuData, Number

logger = logging.getLogger(__name__)


def _psi_absolute_change_fn(scales_1: np.ndarray, scales_2: np.ndarray, pseudocounts) -> np.ndarray:
    """Raw PSI difference (scales_2 - scales_1), used as the DS change function.

    scvi-tools' default change_fn for mode="change" is log2 fold-change, which
    thresholds `delta` as a ratio, not a difference. PSI is a bounded [0, 1]
    proportion reported downstream as a raw difference (`effect_size`), so the
    hypothesis test needs to operate on that same scale for `delta` to mean
    "minimum |ΔPSI|" as intended, rather than "minimum PSI fold-change".
    """
    return scales_2 - scales_1


def _significant_at_fdr(posterior_probabilities: pd.Series, fdr: float) -> pd.Series:
    """Features significant at a target posterior expected FDR, with ties broken by feature order.

    Same rule as scvi-tools' ``_fdr_de_prediction``: sort features by posterior probability (highest first) and keep
    the longest leading run whose mean (1 - probability) is at most ``fdr``. scvi-tools sorts with an unstable sort,
    so when several features share the probability at the cutoff, which of them are kept depends on the CPU's sort
    implementation (it differs between machines with and without AVX-512). A stable sort keeps tied features in their
    input order, so the result is the same on every machine. The number of significant features is unchanged.
    """
    probabilities = posterior_probabilities.to_numpy()
    order = np.argsort(-probabilities, kind="stable")

    sorted_probabilities = probabilities[order]
    cumulative_fdr = np.cumsum(1.0 - sorted_probabilities) / (1.0 + np.arange(len(sorted_probabilities)))
    n_significant = int((cumulative_fdr <= fdr).sum())

    significant = np.zeros(len(probabilities), dtype=bool)
    significant[order[:n_significant]] = True
    return pd.Series(significant, index=posterior_probabilities.index)


def _sort_and_flag_stably(result: pd.DataFrame, feature_names, mode: str, fdr: float) -> pd.DataFrame:
    """Re-sort a ``_de_core`` table and recompute its FDR flags with stable sorts (see :func:`_significant_at_fdr`).

    ``_de_core`` returns each comparison sorted by ``proba_de`` (or ``bayes_factor`` in ``"vanilla"`` mode) with an
    unstable sort. Here each comparison is put back in feature order and sorted again with a stable sort, so that tied
    features keep their feature order, and the FDR flags are recomputed with :func:`_significant_at_fdr`. All other values are
    unchanged.
    """
    sort_column = "proba_de" if mode == "change" else "bayes_factor"
    flag_column = f"is_de_fdr_{fdr}"

    if "comparison" in result.columns:
        comparisons = [table for _, table in result.groupby("comparison", sort=False)]
    else:
        comparisons = [result]

    sorted_comparisons = []
    for table in comparisons:
        table = table.loc[feature_names]
        table = table.iloc[np.argsort(-table[sort_column].to_numpy(), kind="stable")].copy()
        if mode == "change":
            table[flag_column] = _significant_at_fdr(table["proba_de"], fdr)
        sorted_comparisons.append(table)
    return pd.concat(sorted_comparisons, axis=0)

class MyAdvTrainingPlan(AdversarialTrainingPlan):

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        # every other AdversarialTrainingPlan argument is passed through **kwargs
        lr_scheduler_type: Literal["plateau", "step"] = "step",
        step_size: int = 400,
        gradient_clipping: bool = True,
        gradient_clipping_max_norm: float = 5.0,
        cross_gate_mode: str = "hard",
        disable_cross_gate: bool = False,
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)
        self.cross_gate_mode = cross_gate_mode  # "hard" or "soft"
        # If True, the cross gate is pinned open (1.0) for the entire run, including during KL
        # warmup. For non-concatenate modality_weights, this disables the random per-batch
        # single-modality routing in SPLICEVAE.inference() (warmup_only_splicing), so the joint
        # posterior is the real mixed latent from step 1 -- ordinary KL warmup, no modality
        # switching. For modality_weights="concatenate" it also disables the closed-during-warmup
        # cross-term gating. Lets you test a higher splicing_loss_weight without the warmup-routing
        # confound (see splicevi-phi-concentration-findings / gradient-imbalance discussion).
        self.disable_cross_gate = disable_cross_gate
        # new scheduling params
        self.lr_scheduler_type = lr_scheduler_type
        self.step_size = step_size
        self.gradient_clipping = gradient_clipping
        self.gradient_clipping_max_norm = gradient_clipping_max_norm


    def compute_and_log_metrics(self, loss_output, metrics, mode):
        # 1. original ELBO, total recon, total KL, and extra‐metrics
        super().compute_and_log_metrics(loss_output, metrics, mode)

        # 2. now log each modality’s recon loss as the *batch mean*
        for key, val in loss_output.reconstruction_loss.items():
            if isinstance(val, torch.Tensor):
                # val might be a vector of per-cell losses → take mean
                val = val.mean()
            self.log(
                f"{key}_{mode}",
                val,
                on_step=False,
                on_epoch=True,
                batch_size=loss_output.n_obs_minibatch,
                sync_dist=self.use_sync_dist,
            )

        try:
            # PL’s API: self.lr_schedulers() returns a list of dicts
            schedulers = self.lr_schedulers()
            last_lrs = schedulers.get_last_lr()
            current_lr = last_lrs[0] if isinstance(last_lrs, (list, tuple)) else last_lrs
        except Exception:
            # fallback to optimizer if scheduler isn’t available
            optim = self.optimizers()
            optim = optim[0] if isinstance(optim, list) else optim
            current_lr = optim.param_groups[0]["lr"]

        self.log(
            f"lr_{mode}",
            current_lr,
            on_step=True,    # logs every step
            on_epoch=False,  # not aggregated at epoch end
        )

    def _compute_gate(self) -> float:
        if self.disable_cross_gate:
            return 1.0  # pinned open: no warmup routing / cross-term gating at all
        # KL warmup drives self.kl_weight from 0→1
        if self.cross_gate_mode == "soft":
            return float(self.kl_weight)                 # gradually open
        return 1.0 if float(self.kl_weight) >= 0.999 else 0.0  # snap open when warmup done
    
    def on_validation_epoch_start(self):
        if hasattr(self.module, "set_cross_gate"):
            self.module.set_cross_gate(1.0)  # always open for validation

    def on_validation_epoch_end(self) -> None:
        """Update the learning rate via scheduler steps."""
        if self.lr_scheduler_type == "step":
            sch = self.lr_schedulers()
            sch.step()
            return
        
        if (not self.reduce_lr_on_plateau or "validation" not in self.lr_scheduler_metric):
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.lr_scheduler_type == "step":
            logger.info(f"Learning rate: step schedule (x{self.lr_factor} every {self.step_size} epochs)")
            scheduler = StepLR(
                optimizer1,
                step_size=self.step_size,
                gamma=self.lr_factor,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "epoch",
                    },
                },
            )
            
        elif self.reduce_lr_on_plateau and self.lr_scheduler_type == "plateau":
            logger.info("Learning rate: reduced on plateau of the validation loss")
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        if self.adversarial_classifier is not False:
            params2 = filter(lambda p: p.requires_grad, self.adversarial_classifier.parameters())
            optimizer2 = torch.optim.Adam(
                params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
            )
            config2 = {"optimizer": optimizer2}

            # pytorch lightning requires this way to return
            opts = [config1.pop("optimizer"), config2["optimizer"]]
            if "lr_scheduler" in config1:
                scheds = [config1["lr_scheduler"]]
                return opts, scheds
            else:
                return opts

        return config1
    
    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
            self.log("kl_weight", self.kl_weight, on_step=True, on_epoch=False)

        #toggle cross gate right before forward <<<
        if hasattr(self.module, "set_cross_gate"):
            gate = self._compute_gate()
            self.module.set_cross_gate(gate)
            self.log("cross_gate", gate, on_step=True, on_epoch=False)

        kappa = (
            1 - self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        batch_tensor = batch[REGISTRY_KEYS.BATCH_KEY]

        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        z = inference_outputs["z"]
        loss = scvi_loss.loss
        # fool classifier if doing adversarial training
        if kappa > 0 and self.adversarial_classifier is not False:
            fool_loss = self.loss_adversarial_classifier(z, batch_tensor, False)
            loss += fool_loss * kappa

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        opt1.zero_grad()
        self.manual_backward(loss)
        if self.gradient_clipping:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), max_norm=self.gradient_clipping_max_norm)
        opt1.step()

        # train adversarial classifier
        # this condition will not be met unless self.adversarial_classifier is not False
        if opt2 is not None:
            loss = self.loss_adversarial_classifier(z.detach(), batch_tensor, True)
            loss *= kappa
            opt2.zero_grad()
            self.manual_backward(loss)
            if self.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(self.module.parameters(),  max_norm=self.gradient_clipping_max_norm)
            opt2.step()

class SPLICEVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, ArchesMixin):
    """Joint model of gene expression and alternative splicing (junction usage) in single cells.

    SPLICEVI learns one latent space from both modalities: an expression encoder and a splicing encoder each give a
    posterior over the latent space, the two are mixed into a joint posterior, and one decoder per modality
    reconstructs the data (zero-inflated negative binomial for expression, Dirichlet-multinomial over the junctions of
    each ATSE for splicing). It wraps the :class:`~splicevi.splicevae.SPLICEVAE` module.

    The defaults are the settings of the paper's models, so ``SPLICEVI(mdata)`` followed by ``model.train()`` trains
    the paper's model. :meth:`preset` returns the same settings as dictionaries, with the variants for a batch key
    (``atlas="tms"``) and for a reference plus a sparse atlas (``"sparse"``, ``joint=True``). 

    Parameters
    ----------
    adata
        MuData object with the modalities ``"rna"`` and ``"splicing"``, registered with :meth:`setup_mudata`.
    n_genes
        Number of genes. Taken from the registered data when left as None.
    n_junctions
        Number of junctions. Taken from the registered data when left as None.
    modality_weights
        How the two posteriors are mixed into the joint latent:

        * ``"equal"`` (default): equal weight per modality,
        * ``"universal"``: one learned weight per modality,
        * ``"cell"``: learned weights per cell,
        * ``"concatenate"``: no mixing; the two latents are concatenated,
        * ``"per_dimension_weighted_average"``: one learned splicing weight per latent dimension (expression weight =
          1 - splicing weight), constrained so that the mean weight stays at 0.5.
    modality_penalty
        Penalty that aligns the two posteriors: ``"None"`` (default), ``"Jeffreys"`` or ``"MMD"``.
    n_hidden
        Width of the hidden layers of the encoders and decoders. If None (default), the square root of the number of
        junctions, capped at 128.
    n_latent
        Dimension of the latent space (of each modality's latent before concatenation). Default 30. If None, the
        square root of ``n_hidden``.
    n_layers_encoder
        Hidden layers in the encoders, including the network after pooling in the partial splicing encoder. Default 2.
    n_layers_decoder
        Hidden layers in the decoders (not used by the linear decoders). Default 2.
    dropout_rate
        Dropout rate in the hidden layers. Default 0.01.
    use_batch_norm
        Where to apply batch normalization: ``"encoder"``, ``"decoder"``, ``"both"`` or ``"none"`` (default).
    use_layer_norm
        Where to apply layer normalization: ``"encoder"``, ``"decoder"``, ``"both"`` (default) or ``"none"``.
    latent_distribution
        Latent distribution: ``"normal"`` (default) or ``"ln"`` (logistic normal).
    deeply_inject_covariates
        If True, covariates enter every decoder layer. Default False.
    encode_covariates
        If True, covariates are also given to the encoders. Default False.
    gene_likelihood
        Expression likelihood: ``"zinb"`` (default), ``"nb"`` or ``"poisson"``.
    dispersion
        Expression dispersion: ``"gene"`` (default), ``"gene-batch"``, ``"gene-label"`` or ``"gene-cell"``.
    fully_paired
        If True, the model exposes only the joint latent (no latent per modality). Default False.
    **model_kwargs
        Passed on to :class:`~splicevi.splicevae.SPLICEVAE`.

    Other Parameters
    ----------------
    variance_mixing
        How the two encoders' variances are combined into the joint posterior variance (every ``modality_weights``
        except ``"concatenate"``; the joint mean is always the weighted average of the two means):

        * ``"squared"`` (default): sum(w**2 * v), the variance of a weighted average of independent Gaussians,
        * ``"sqrt_weights"``: for equal/cell/universal weights the MultiVI rule sum(sqrt(w) * v) (about 1.41 times the
          average variance under equal weights); for ``per_dimension_weighted_average`` (sum(w * std))**2,
        * ``"linear"``: sum(w * v).
    mask_cells_without_splicing
        If True, cells with no observed junction get no splicing weight in the joint latent and are left out of the
        alignment penalty. Default False: every cell counts as having splicing.
    splicing_loss_type
        Splicing likelihood: ``"dirichlet_multinomial"`` (default), ``"beta_binomial"`` or ``"binomial"``.
    dm_concentration
        Dirichlet-multinomial concentration: ``"atse"`` (default; one learned concentration per ATSE) or ``"scalar"``
        (one shared value).
    splicing_loss_weight
        Multiplier of the splicing reconstruction loss relative to the expression reconstruction loss. Default 1.0.
    lambda_prior
        Weight of the L2 penalty that pulls each learned log-concentration toward 0 (``phi_prior="l2_log"`` only).
        Default 0.01; 0.0 turns it off.
    phi_floor
        Minimum concentration: ``phi = phi_floor + softplus(log_phi)``. Default 0.0.
    phi_prior
        Regularizer on the concentration: ``"l2_log"`` (default; see ``lambda_prior``), ``"gamma"`` (Gamma prior on
        phi with ``phi_prior_shape`` and ``phi_prior_rate``, scaled by 1 / number of cells) or ``"none"``.
    phi_prior_shape, phi_prior_rate
        Shape and rate of the Gamma prior. Defaults 2.0 and 0.1 (mean 20).
    phi_init
        Starting concentration: ``"constant"`` (default; every ATSE starts at the same value, phi about 4.6),
        ``"log100"`` (the same start plus a random draw per ATSE) or ``"prior"`` (a draw from the Gamma prior).
    psi_prior_pseudocount
        Strength of the pseudo-count term on the decoded junction usage: adds
        ``-psi_prior_pseudocount * sum_j pop_psi_j * log(psi_j)`` per cell, i.e. that many virtual reads per ATSE
        distributed as the read-weighted population usage of the training data. Dirichlet-multinomial only.
        Default 0.0 (off).
    splicing_encoder_architecture
        Splicing encoder: ``"partial"`` (default; reads only the observed junctions of each cell) or ``"vanilla"``
        (scvi-tools ``Encoder`` on the full junction vector).
    splicing_decoder_architecture
        Splicing decoder: ``"vanilla"`` (default; nonlinear) or ``"linear"``.
    expression_architecture
        Expression decoder: ``"vanilla"`` (default; nonlinear) or ``"linear"``.
    encoder_hidden_dim
        Width of the network after pooling in the partial splicing encoder. Default 128.
    code_dim
        Size of the learned embedding of each junction in the partial splicing encoder. Default 64.
    h_hidden_dim
        Width of the network that combines each observed junction's input with its embedding. Default 128.
    pool_mode
        Pooling over the observed junctions of a cell: ``"mean"`` (default) or ``"sum"``.
    psi_input
        Input of the partial splicing encoder: ``"deviation"`` (default; each observed junction's usage minus its
        read-weighted population usage, pooled with a fixed divisor so the pooled code does not carry the number of
        observed junctions; needs ``pool_mode="mean"``) or ``"raw"`` (the observed usage itself).
    decoder_intercept_init
        Starting bias of the splicing decoder: ``"population"`` (default; set so that the decoder starts at the
        read-weighted population usage of the training data) or ``"default"`` (PyTorch's initialization).
    decoder_depth_covariates
        If True (default), both decoders (never the encoders) receive three standardized depth inputs per cell: log1p
        detected genes, log1p observed junctions and log1p library size. Vanilla splicing decoder only.
    max_nobs
        Largest number of observed (cell, junction) pairs processed in one chunk by the partial encoder; a negative
        value (default -1) processes a batch in one chunk.
    n_decoder_covariates
        Number of zero-initialized covariate inputs into the first hidden layer of both decoders. Default 0. Normally
        left at 0 and added by :meth:`finetune_decoders` (one input, technology); a saved fine-tuned model stores it,
        so :meth:`load` rebuilds the inputs. Vanilla decoders only.
    initialize_embeddings_from_pca
        If True (default) and the splicing encoder is ``"partial"``, the junction embeddings start from a truncated SVD
        of the junction usage matrix.

    Examples
    --------
    >>> SPLICEVI.setup_mudata(mdata)   # the paper's layer names; add batch_key if there is one
    >>> model = SPLICEVI(mdata)
    >>> model.train()
    >>> latent = model.get_latent_representation()
    """
    _module_cls = SPLICEVAE
    _training_plan_cls = MyAdvTrainingPlan

    def __init__(
        self,
        adata: AnnOrMuData,
        n_genes: int | None = None,
        n_junctions: int | None = None,

        # --- Modality mixing ---
        modality_weights: Literal["equal", "cell", "universal", "concatenate", "per_dimension_weighted_average"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "None",
        variance_mixing: Literal["sqrt_weights", "linear", "squared"] = "squared",
        mask_cells_without_splicing: bool = False,

        # --- Shared SCVI-style encoder/decoder hyperparameters ---
        n_hidden: int | None = None,
        n_latent: int | None = 30,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.01,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        encode_covariates: bool = False,

        # --- Likelihoods & dispersion (expression) ---
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",

        # --- Splicing likelihood ---
        splicing_loss_type: Literal["binomial", "beta_binomial", "dirichlet_multinomial"] = "dirichlet_multinomial",
        dm_concentration: Literal["atse", "scalar"] = "atse",
        splicing_loss_weight: float = 1.0,
        lambda_prior: float = 1e-2,
        phi_floor: float = 0.0,
        phi_prior: Literal["l2_log", "gamma", "none"] = "l2_log",
        phi_prior_shape: float = 2.0,
        phi_prior_rate: float = 0.1,
        phi_init: Literal["log100", "prior", "constant"] = "constant",
        psi_prior_pseudocount: float = 0.0,

        # --- Architecture toggles ---
        splicing_encoder_architecture: Literal["vanilla", "partial"] = "partial",
        splicing_decoder_architecture: Literal["vanilla", "linear"] = "vanilla",
        expression_architecture: Literal["vanilla", "linear"] = "vanilla",

        # --- PartialEncoder knobs (splicing_architecture="partial") ---
        encoder_hidden_dim: int = 128,
        code_dim: int = 64,
        h_hidden_dim: int = 128,
        pool_mode: Literal["mean", "sum"] = "mean",
        psi_input: Literal["raw", "deviation"] = "deviation",                    # splicing-encoder input: psi - population psi, h(0) subtracted, fixed pool divisor
        decoder_intercept_init: Literal["default", "population"] = "population",  # splicing-decoder bias starts at log(read-weighted population psi)
        decoder_depth_covariates: bool = True,                                   # depth/detection covariates into both decoders
        max_nobs: int = -1,
        n_decoder_covariates: int = 0,                                        # decoder covariate inputs (e.g. technology); set by finetune_decoders

        # --- Model-only helpers ---
        initialize_embeddings_from_pca: bool = True,
        fully_paired: bool = False,

        # (module bookkeeping inferred from data managers)
        **model_kwargs,
    ):
        super().__init__(adata)

        # Numbers of genes and junctions: taken from the registered modalities when not given.
        if n_genes is None or n_junctions is None:
            if not isinstance(adata, MuData):
                raise ValueError("n_genes and n_junctions must be provided if using AnnData")
            registry = self.adata_manager.data_registry
            if n_genes is None:
                n_genes = self.adata[registry[REGISTRY_KEYS.X_KEY].mod_key].n_vars
            if n_junctions is None:
                n_junctions = self.adata[registry["junc_ratio"].mod_key].n_vars

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else []
        )
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry

        self.module = self._module_cls(
            n_input_genes=n_genes,
            n_input_junctions=n_junctions,
            modality_weights=modality_weights,
            modality_penalty=modality_penalty,
            variance_mixing=variance_mixing,
            mask_cells_without_splicing=mask_cells_without_splicing,
            n_batch=self.summary_stats.n_batch,
            n_obs=adata.n_obs,
            n_labels=self.summary_stats.get("n_labels", 0),

            # shared SCVI-style
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            n_layers_decoder=n_layers_decoder,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            encode_covariates=encode_covariates,
            use_size_factor_key=use_size_factor_key,

            # likelihoods
            gene_likelihood=gene_likelihood,
            gene_dispersion=dispersion,
            splicing_loss_type=splicing_loss_type,
            dm_concentration=dm_concentration,
            splicing_loss_weight=splicing_loss_weight,
            lambda_prior=lambda_prior,
            phi_floor=phi_floor,
            phi_prior=phi_prior,
            phi_prior_shape=phi_prior_shape,
            phi_prior_rate=phi_prior_rate,
            phi_init=phi_init,
            psi_prior_pseudocount=psi_prior_pseudocount,

            # architectures
            splicing_encoder_architecture=splicing_encoder_architecture,
            splicing_decoder_architecture=splicing_decoder_architecture,
            expression_architecture=expression_architecture,

            # partial encoder knobs
            encoder_hidden_dim=encoder_hidden_dim,
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            pool_mode=pool_mode,
            psi_input=psi_input,
            decoder_depth_covariates=decoder_depth_covariates,
            max_nobs=max_nobs,
            n_decoder_covariates=n_decoder_covariates,

            # extras
            **model_kwargs,
        )

        self._model_summary_string = (
            f"SpliceVI | genes={n_genes}, junctions={n_junctions} | "
            f"n_hidden={self.module.n_hidden}, n_latent={self.module.n_latent}, "
            f"enc_layers={n_layers_encoder}, dec_layers={n_layers_decoder}, "
            f"dropout={dropout_rate}, z_dist={latent_distribution} | "
            f"expr_dec={expression_architecture}, spl_enc={splicing_encoder_architecture}, "
            f"spl_dec={splicing_decoder_architecture} | "
            f"gene_like={gene_likelihood}, disp={dispersion} | "
            f"splicing_loss={splicing_loss_type}, dm_conc={dm_concentration}, "
            f"sp_loss_weight={splicing_loss_weight}, "
            f"lambda_prior={lambda_prior}, phi_prior={phi_prior}, phi_floor={phi_floor}, phi_init={phi_init}, "
            f"psi_prior_pseudocount={psi_prior_pseudocount} | "
            f"mix={modality_weights}, penalty={modality_penalty}, var_mix={variance_mixing}, mask_no_spl={mask_cells_without_splicing} | "
            f"PE(code_dim={code_dim}, h_hidden={h_hidden_dim}, "
            f"enc_hidden={encoder_hidden_dim}, pool={pool_mode}, "
            f"max_nobs={max_nobs}, psi_input={psi_input}) | "
            f"init_from_pca={initialize_embeddings_from_pca}, decoder_intercept_init={decoder_intercept_init}, "
            f"decoder_depth_covariates={decoder_depth_covariates}"
        )

        self.fully_paired = fully_paired
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        self.n_genes = n_genes
        self.n_junctions = n_junctions
        self.get_normalized_function_name = "get_normalized_splicing"
        self.dm_concentration = dm_concentration
        self.lambda_prior = lambda_prior

        if self.adata is not None:
            if initialize_embeddings_from_pca and splicing_encoder_architecture == "partial":
                self.init_feature_embedding_from_adata()
            if psi_input == "deviation" and splicing_encoder_architecture == "partial":
                self.init_pop_psi_from_adata()
            if psi_input == "deviation" and splicing_encoder_architecture == "partial":
                self.init_pool_divisor_from_adata()
            if decoder_intercept_init == "population" and splicing_decoder_architecture == "vanilla":
                self.init_decoder_intercept_from_adata()
            if decoder_depth_covariates:
                self.init_depth_covariate_stats_from_adata()

            # Only needed for DM splicing likelihood
            if splicing_loss_type == "dirichlet_multinomial":
                self.init_junc2atse()
            if psi_prior_pseudocount > 0:
                self.init_psi_prior_from_adata()

    def make_junc2atse(self, atse_labels):
        """Sparse (junctions x ATSEs) matrix with a 1 where a junction belongs to an ATSE.

        With ``dm_concentration="atse"`` this also replaces the concentration parameter by one value per ATSE.
        """
        num_junctions = len(atse_labels)
        atse_labels = atse_labels.astype('category')
        row_indices = torch.arange(num_junctions, dtype=torch.long)
        col_indices = torch.tensor(atse_labels.cat.codes.values)
        num_atses = len(atse_labels.cat.categories)
        logger.info(f"Found {num_junctions} junctions in {num_atses} ATSEs.")

        if self.dm_concentration == "atse":
            self.module.log_phi_j = self.module.init_log_phi(num_atses)
            self.module.log_phi_j.requires_grad_(True)

        return torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=torch.ones(len(row_indices), dtype=torch.float32),
            size=(num_junctions, len(atse_labels.cat.categories))
        ).coalesce()

    def init_junc2atse(self) -> None:
        """Build the junction-to-ATSE matrix from ``var["event_id"]`` of the splicing modality (once)."""
        if hasattr(self.module, "junc2atse"):
            return

        atse_counts_info = self.adata_manager.data_registry["atse_counts_key"]
        splicing_data = self.adata[atse_counts_info.mod_key]
        if atse_counts_info.attr_key not in splicing_data.layers:
            raise KeyError(f"ATSE count layer {atse_counts_info.attr_key!r} not found in the splicing modality.")
        atse_labels = splicing_data.var["event_id"]

        junction_to_atse = self.make_junc2atse(atse_labels)
        self.module.junc2atse = junction_to_atse.coalesce().to(self.module.device)

    def population_psi_from_adata(self) -> np.ndarray:
        """Read-weighted population PSI per junction from the registered count layers of the training data.

        Sum over cells of the junction counts divided by the sum over cells of the ATSE totals (the ATSE count layer
        repeats the ATSE total on every junction of the ATSE, so its column sum is the ATSE's read total); 0 for
        junctions without reads.
        """
        junction_counts_info = self.adata_manager.data_registry["junc_counts_key"]
        atse_counts_info = self.adata_manager.data_registry["atse_counts_key"]
        junction_counts = self.adata[junction_counts_info.mod_key].layers[junction_counts_info.attr_key]
        atse_counts = self.adata[atse_counts_info.mod_key].layers[atse_counts_info.attr_key]

        junction_reads = np.asarray(junction_counts.sum(axis=0)).ravel().astype(np.float64)
        atse_reads = np.asarray(atse_counts.sum(axis=0)).ravel().astype(np.float64)
        return np.divide(junction_reads, atse_reads, out=np.zeros_like(junction_reads), where=atse_reads > 0)

    def init_pop_psi_from_adata(self) -> None:
        """Fill the partial encoder's ``pop_psi`` buffer with the population PSI of the training data
        (``psi_input="deviation"``)."""
        p_pop = self.population_psi_from_adata()
        with torch.no_grad():
            buf = self.module.z_encoder_splicing.pop_psi
            buf.copy_(torch.as_tensor(p_pop, dtype=buf.dtype, device=buf.device))
        logger.info(f"Population PSI set for the encoder input: {int((p_pop > 0).sum())} of {len(p_pop)} junctions "
                    f"with reads; mean {p_pop.mean():.4f}")

    def init_pool_divisor_from_adata(self) -> None:
        """Set the partial encoder's fixed pooling divisor (``psi_input="deviation"``): the median number of observed
        junctions per training cell."""
        if "psi_observed_mask" in self.adata_manager.data_registry:
            observed_mask = self.adata_manager.get_from_registry("psi_observed_mask")
        else:
            observed_mask = self.adata["splicing"].layers["psi_mask"]
        observed_per_cell = np.asarray((sp.csr_matrix(observed_mask) > 0).sum(1)).ravel()
        median_observed = float(np.median(observed_per_cell))
        with torch.no_grad():
            self.module.z_encoder_splicing.pool_divisor.fill_(max(median_observed, 1.0))
        logger.info(f"Pooling divisor set: median observed junctions per training cell = {median_observed:.0f}")

    def init_decoder_intercept_from_adata(self) -> None:
        """Start the splicing decoder's per-junction bias at log(population PSI), clipped to [1e-4, 1].

        With the hidden contribution at zero, the decoder then reproduces the population PSI: ``generative()`` maps
        sigmoid(bias) -> clamp -> logit -> softmax within the ATSE, which is the softmax of the bias over the ATSE.
        """
        p_pop = self.population_psi_from_adata()
        with torch.no_grad():
            b = self.module.z_decoder_splicing.ps_output.bias
            b.copy_(torch.as_tensor(np.log(np.clip(p_pop, 1e-4, 1.0)), dtype=b.dtype, device=b.device))
        logger.info(f"Splicing decoder bias set from the population PSI ({len(p_pop)} junctions)")

    def init_psi_prior_from_adata(self) -> None:
        """Population PSI for the pseudo-count prior (psi_prior_pseudocount > 0): read-weighted population PSI per
        junction; junctions of ATSEs with no reads in the training data get 1 / (junctions in the ATSE), so the prior
        is a distribution within every ATSE."""
        p_pop = self.population_psi_from_adata()
        splicing_mod_key = self.adata_manager.data_registry["junc_counts_key"].mod_key
        atse_codes = pd.factorize(self.adata[splicing_mod_key].var["event_id"].astype(str))[0]
        atse_size = np.bincount(atse_codes)[atse_codes]
        no_reads = np.bincount(atse_codes, weights=p_pop)[atse_codes] == 0
        p_pop = np.where(no_reads, 1.0 / atse_size, p_pop)
        with torch.no_grad():
            self.module.psi_prior_pop.copy_(torch.as_tensor(p_pop, dtype=self.module.psi_prior_pop.dtype))
        logger.info(f"PSI prior: population PSI set for {len(p_pop)} junctions ({int(no_reads.sum())} in ATSEs "
                    f"without reads -> uniform); pseudo-count {self.module.psi_prior_pseudocount}")

    def init_depth_covariate_stats_from_adata(self) -> None:
        """Mean and standard deviation over the training cells of [log1p detected genes, log1p observed junctions,
        log1p library size], stored as buffers so the decoders' depth covariates are standardized with training
        constants (same definitions as in ``SPLICEVAE.inference``)."""
        registry = self.adata_manager.data_registry
        rna_info = registry[REGISTRY_KEYS.X_KEY]
        mask_info = registry["psi_observed_mask"]

        rna_data = self.adata[rna_info.mod_key]
        expression = rna_data.layers[rna_info.attr_key] if rna_info.attr_name == "layers" else rna_data.X
        detected_genes = np.asarray((expression > 0).sum(axis=1)).ravel().astype(np.float64)

        observed_mask = self.adata[mask_info.mod_key].layers[mask_info.attr_key]
        observed_junctions = np.asarray((observed_mask > 0).sum(axis=1)).ravel().astype(np.float64)

        # The size factor is a joint obsm field of the MuData (no mod_key), so it is read through the manager.
        if REGISTRY_KEYS.SIZE_FACTOR_KEY in registry:
            library_size = self.adata_manager.get_from_registry(REGISTRY_KEYS.SIZE_FACTOR_KEY)
            library_size = np.asarray(library_size).ravel().astype(np.float64)
        else:
            library_size = np.asarray(expression.sum(axis=1)).ravel()

        covariates = np.log1p(np.column_stack([detected_genes, observed_junctions, library_size]))
        with torch.no_grad():
            mean_buffer, std_buffer = self.module.depth_cov_mean, self.module.depth_cov_std
            mean_buffer.copy_(torch.as_tensor(covariates.mean(0), dtype=mean_buffer.dtype))
            std_buffer.copy_(torch.as_tensor(covariates.std(0) + 1e-6, dtype=std_buffer.dtype))
        logger.info(f"Depth covariate statistics set from the training cells: mean {covariates.mean(0).round(3)}, "
                    f"std {covariates.std(0).round(3)}")

    def init_feature_embedding_from_adata(self) -> None:
        """Start the partial encoder's junction embeddings from a truncated SVD of the junction-usage matrix."""
        jr_info = self.adata_manager.data_registry["junc_ratio"]
        jr_key, mod_key = jr_info.attr_key, jr_info.mod_key
        X = self.adata[mod_key].layers[jr_key]
        # keep X sparse:
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        # Fit a code_dim-component SVD on the sparse junction-usage matrix. A small dataset may not support code_dim
        # components; then only the first columns of the embedding start from the SVD and the rest keep their random start.
        code_dim = self.module.z_encoder_splicing.code_dim
        n_components = min(code_dim, X.shape[0] - 1, X.shape[1] - 1)
        svd = TruncatedSVD(n_components=n_components,
                        algorithm="randomized")
        svd.fit(X)

        # components_.shape == (n_components, n_vars)
        comps = svd.components_.T  # (n_vars, n_components)

        with torch.no_grad():
            emb = self.module.z_encoder_splicing.feature_embedding
            emb[:, :n_components].copy_(torch.as_tensor(comps, dtype=emb.dtype, device=emb.device))
        if n_components < code_dim:
            logger.info(f"Started {n_components} of {code_dim} junction-embedding columns from the SVD (small dataset).")

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 300,
        lr: float = 1e-4,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 256,
        weight_decay: float = 1e-4,
        eps: float = 1e-08,
        early_stopping: bool = True,
        early_stopping_patience: int = 50,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 200,
        adversarial_mixing: bool = True,
        lr_scheduler_type: Literal["plateau", "step"] = "step",
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.1,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        step_size: int = 400,
        gradient_clipping: bool = True,
        gradient_clipping_max_norm: float = 5.0,
        disable_cross_gate: bool = False,
        num_workers: int = 0,
        pin_memory: bool = False,
        load_sparse_tensor: bool = False,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        r"""Train the model with amortized variational inference on both modalities.

        The defaults are the paper's training settings: 300 epochs, batch size 256, AdamW with learning rate 1e-4 and
        weight decay 1e-4, KL warm-up over 200 epochs, early stopping on the validation reconstruction loss with
        patience 50, and gradient clipping at norm 5. The step scheduler with ``step_size=400`` never lowers the
        learning rate within 300 epochs, so the learning rate stays constant.

        Parameters
        ----------
        max_epochs
            Number of epochs. Default 300.
        lr
            Learning rate. Default 1e-4.
        accelerator, devices
            Hardware acceleration options.
        train_size, validation_size
            Proportions for splitting the data into train and validation sets.
        shuffle_set_split
            Whether to shuffle before splitting.
        batch_size
            Minibatch size. Default 256.
        weight_decay, eps
            AdamW weight decay (default 1e-4) and epsilon (default 1e-8).
        early_stopping
            Whether to stop early on the validation reconstruction loss. Default True.
        early_stopping_patience
            Number of epochs with no improvement before stopping. Default 50.
        check_val_every_n_epoch
            How often (in epochs) to run validation.
        n_steps_kl_warmup, n_epochs_kl_warmup
            KL warm-up length, in steps or in epochs (default 200 epochs).
        adversarial_mixing
            Whether to train an adversarial batch classifier on the latent space. Default True; it is switched off
            automatically when the data has a single batch (no batch key), as for the paper's Allen models, and is on
            with a batch key, as for the paper's TMS models (batch key ``mouse.id``).
        lr_scheduler_type
            Learning-rate scheduler: ``"step"`` (default; multiply by ``lr_factor`` every ``step_size`` epochs) or
            ``"plateau"`` (reduce on plateau).
        reduce_lr_on_plateau
            If True and using plateau scheduler, enable ReduceLROnPlateau.
        lr_factor
            Factor by which the learning rate is multiplied at each reduction (both schedulers). Default 0.1.
        lr_patience
            Number of epochs with no improvement for plateau scheduler.
        lr_threshold
            Threshold for measuring new optimum (plateau scheduler).
        lr_scheduler_metric
            Metric to monitor for plateau scheduler.
        step_size
            Epochs between learning-rate reductions (step scheduler). Default 400.
        gradient_clipping
            Whether or not (true or false) to use gradient norm clipping
        gradient_clipping_max_norm
            Max norm of the gradients to be used in gradient clipping
        disable_cross_gate
            If True, pin the cross gate open (1.0) for the whole run, including during KL warmup.
            For non-concatenate ``modality_weights`` this disables the random per-batch
            single-modality routing (SPLICEVAE.inference()'s warmup_only_splicing), so the joint
            posterior is the real mixed latent from the start -- ordinary KL warmup, no modality
            switching. For ``modality_weights="concatenate"`` it also disables the
            closed-during-warmup cross-term gating. Default False.
        num_workers
            Data-loader worker processes for the training and validation loaders (kept alive across epochs). Batches
            are built in parallel with the GPU step instead of on the main process. Default 0 (main process).
        pin_memory
            Page-locked host memory for batches, which makes the copy to the GPU faster. Default False.
        load_sparse_tensor
            Send sparse layers to the GPU as sparse tensors and make them dense there (scvi-tools'
            ``on_after_batch_transfer``), instead of building dense 79,768-wide matrices on the CPU for every batch.
            Registered splicing layers stored dense are converted to CSR in memory first (same values). The batches
            the model sees are identical. Default False.
        datasplitter_kwargs
            Additional kwargs for the data splitter.
        plan_kwargs
            Additional kwargs to pass to the TrainingPlan constructor.
        **kwargs
            Additional Trainer kwargs (callbacks, strategy, etc.).
        """         
        update_dict = {
            "lr": lr,
            "lr_scheduler_type": lr_scheduler_type,
            "reduce_lr_on_plateau": reduce_lr_on_plateau, 
            "lr_factor": lr_factor,
            "lr_patience": lr_patience,
            "lr_threshold": lr_threshold,
            "lr_scheduler_metric": lr_scheduler_metric,
            "step_size": step_size,
            "adversarial_classifier": adversarial_mixing,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "scale_adversarial_loss": 1,
            "gradient_clipping": gradient_clipping,
            "gradient_clipping_max_norm": gradient_clipping_max_norm,
            "disable_cross_gate": disable_cross_gate,    
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = dict(datasplitter_kwargs or {})
        if num_workers > 0:
            datasplitter_kwargs.setdefault("num_workers", int(num_workers))
            datasplitter_kwargs.setdefault("persistent_workers", True)
        if pin_memory:
            datasplitter_kwargs.setdefault("pin_memory", True)
        if load_sparse_tensor:
            self._sparsify_registered_layers()
            datasplitter_kwargs.setdefault("load_sparse_tensor", True)

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            distributed_sampler=use_distributed_sampler(kwargs.get("strategy", None)),
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_warmup_epochs= n_epochs_kl_warmup,
            early_stopping_patience=early_stopping_patience,
            **kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_library_size_factors(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
    ) -> dict[str, np.ndarray]:
        r"""Return library size factors for gene expression.

        Note that the splicing modality does not use library size factors.

        Parameters
        ----------
        adata
            AnnOrMuData object.
        indices
            Cell indices (default: all cells).
        batch_size
            Batch size for processing.

        Returns
        -------
        A dictionary with key "expression" for the gene expression library size factors.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        lib_exp = []
        for tensors in scdl:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            lib_exp.append(outputs["libsize_expr"].cpu())
        return {"expression": torch.cat(lib_exp).numpy().squeeze()}

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnOrMuData | None = None,
        modality: Literal["joint", "expression", "splicing"] = "joint",
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        r"""Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnOrMuData object used in setup.
        modality
            One of:
              - ``"joint"``: joint latent space,
              - ``"expression"``: expression-specific latent space,
              - ``"splicing"``: splicing-specific latent space.
        indices
            Cell indices to use.
        give_mean
            If True, returns the mean of the latent distribution.
        batch_size
            Batch size for processing.

        Returns
        -------
        A NumPy array of the latent representations.
        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")
        self._check_adata_modality_weights(adata)
        keys = {"z": "z", "qz_m": "qz_m", "qz_v": "qz_v"}
        if self.fully_paired and modality != "joint":
            raise RuntimeError("A fully paired model only has a joint latent representation.")
        if not self.fully_paired and modality != "joint":
            if modality == "expression":
                keys = {"z": "z_expr", "qz_m": "qzm_expr", "qz_v": "qzv_expr"}
            elif modality == "splicing":
                keys = {"z": "z_spl", "qz_m": "qzm_spl", "qz_v": "qzv_spl"}
            else:
                raise RuntimeError("Modality must be 'joint', 'expression', or 'splicing'.")

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs[keys["qz_m"]]
            qz_v = outputs[keys["qz_v"]]
            z = outputs[keys["z"]]
            if give_mean:
                if self.module.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([1])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz_m
            latent.append(z.cpu())
        return torch.cat(latent).numpy()

    def _sparsify_registered_layers(self) -> None:
        """Convert registered layers stored as dense arrays to CSR in memory (same values), so they can be loaded as
        sparse tensors (train(load_sparse_tensor=True)). Expression and splicing layers only; obs/obsm fields untouched."""
        for key, info in self.adata_manager.data_registry.items():
            if info.attr_name != "layers":
                continue
            ad_ = self.adata[info.mod_key] if info.mod_key is not None else self.adata
            X = ad_.layers[info.attr_key]
            if isinstance(X, np.ndarray):
                ad_.layers[info.attr_key] = sp.csr_matrix(X)
                logger.info(f"{info.mod_key}.layers['{info.attr_key}'] ({key}) converted to CSR for sparse loading")

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
        silent: bool = True,
        expected: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnOrMuData object with equivalent structure to initial AnnData. If `None`, defaults
            to the AnnOrMuData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of observations to sample from ``indices`` if ``indices`` is provided.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        use_z_mean
            If True, use the mean of the latent distribution, otherwise sample from it
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a numpy array instead of a pandas DataFrame.
        silent
            If True, disables the progress bar.
        expected
            If True, return the ZINB expected expression :math:`(1 - \pi) \rho_n`, with :math:`\pi` the decoder's
            dropout probability, instead of :math:`\rho_n`. Only for ``gene_likelihood="zinb"``. Used by DE with
            ``norm_expression_function="zinb_expected"``.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is
        :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if expected and getattr(self.module, "gene_likelihood", None) != "zinb":
            raise ValueError("expected=True requires gene_likelihood='zinb' (with NB, px_scale already is the expected expression).")
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names[: self.n_genes]
            gene_mask = [gene in gene_list for gene in all_genes]

        exprs = []
        for tensors in scdl:
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                if batch is not None:
                    batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(batch_indices) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                output = generative_outputs["px_scale"]
                if expected:
                    output = output * (1 - torch.sigmoid(generative_outputs["px_dropout"]))
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                per_batch_exprs.append(output)
            per_batch_exprs = np.stack(
                per_batch_exprs
            )  # shape is (len(transform_batch) x batch_size x n_var)
            exprs += [per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy:
            return exprs
        else:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[: self.n_genes][gene_mask],
                index=adata.obs_names[indices],
            )

    @torch.inference_mode()
    def get_normalized_splicing(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        junction_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
        silent: bool = True,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized (decoded) splicing probabilities.

        This is denoted as :math:`p_{nj}` in the SPLICEVI model.

        Parameters
        ----------
        adata
            AnnOrMuData object with the same structure as used in setup. If `None`,
            defaults to the AnnOrMuData object used to initialize the model.
        indices
            Cell indices to use. If `None`, all cells are used.
        n_samples_overall
            Number of observations to sample from `indices` if provided.
        transform_batch
            Batch(s) to condition on:
            - None: use true observed batch
            - int: force all cells to that batch
            - list[int|str]: average over those batches
        junction_list
            Subset of junction names to return. If `None`, returns all junctions.
        use_z_mean
            If True, use the mean of the latent distribution; otherwise sample.
        n_samples
            Number of posterior samples to draw. If >1 and `return_mean` is True,
            the result is averaged over draws.
        batch_size
            Minibatch size for decoding. Defaults to `scvi.settings.batch_size`.
        return_mean
            Whether to average over posterior samples when `n_samples>1`.
        return_numpy
            If True, returns a NumPy array; otherwise a pandas DataFrame.
        silent
            If True, suppresses the progress bar.

        Returns
        -------
        A NumPy array or pandas DataFrame of shape `(cells, junctions)` containing the
        decoded splicing probabilities.
        """
        # Validate and prepare
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        # Select cells
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, size=n_samples_overall)

        # Data loader
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        # Batches to transform over
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        # Build junction mask
        all_junc = adata.var_names[self.n_genes : self.n_genes + self.n_junctions]
        if junction_list is None:
            junction_mask = slice(None)
        else:
            junction_mask = [j in junction_list for j in all_junc]

        # Decode in mini-batches
        spls = []
        for tensors in scdl:
            per_batch_spls = []
            for batch in track(transform_batch, disable=silent):
                if batch is not None:
                    # override batch
                    bidx = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(bidx) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                output = generative_outputs["p"]
                output = output[..., junction_mask]
                output = output.cpu().numpy()
                per_batch_spls.append(output)
            per_batch_spls = np.stack(per_batch_spls)  # (len(transform_batch), bs, n_junc)
            spls += [per_batch_spls.mean(0)]

        # Concatenate across minibatches
        spls = np.concatenate(spls, axis=0)  # (n_cells, n_selected_junc)

        # If multiple samples requested and we averaged over them in forward
        if n_samples > 1 and return_mean:
            spls = spls.mean(0)

        if return_numpy:
            return spls

        # Build DataFrame
        cols = (
            all_junc
            if junction_list is None
            else [j for j, keep in zip(all_junc, junction_mask) if keep]
        )
        return pd.DataFrame(spls, index=adata.obs_names[indices], columns=cols)

    @torch.inference_mode()
    def get_normalized_splicing_DM(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        junction_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
        silent: bool = True,
        sample: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns **DM-normalized** splicing probabilities (PSI*).

        This decodes junction PSI (:math:`p`) and applies a Dirichlet–multinomial
        posterior mean smoothing per junction:
        :math:`\psi^\* = (c p + y_j) / (c + n)`,
        where :math:`y_j` is the observed junction count and :math:`n` is the
        observed ATSE total for that junction. The concentration :math:`c` is
        taken from the module:
        * If ``self.dm_concentration == "atse"``: use the per-ATSE values in
            ``self.module.log_phi_j`` mapped to per-junction via
            ``self.module.junc2atse``.
        * Otherwise: treat ``self.module.log_phi_j`` as a scalar concentration.

        Parameters
        ----------
        adata
            AnnOrMuData object with equivalent structure to initialization.
            If `None`, defaults to the object used to initialize the model.
        indices
            Indices of cells in `adata` to use. If `None`, all cells are used.
        n_samples_overall
            Number of observations to sample from ``indices`` if provided.
        transform_batch
            Batch conditioning:
            - None: use observed batch
            - int/str: force decode to that batch
            - list[int|str]: average decoded outputs over listed batches
        junction_list
            Optional subset of junction names to return.
        use_z_mean
            If True, decode from the mean of the latent; else sample.
        n_samples
            Posterior samples to draw. If >1 and `return_mean` is True, results are
            averaged over samples.
        batch_size
            Minibatch size for decoding.
        return_mean
            If True and `n_samples` > 1, average samples before returning.
        return_numpy
            If True, return a NumPy array; else a pandas DataFrame.
        silent
            If True, disables progress display.
        sample
            If True, return one random draw from the per-junction posterior
            :math:`\psi_j \sim \mathrm{Beta}(c p_j + y_j,\ c (1 - p_j) + n - y_j)` (the Beta marginal of the
            Dirichlet posterior, whose mean is :math:`\psi^\*`) instead of its mean. Used by DS with
            ``norm_splicing_function="dm_posterior_sample"``: junctions with few reads get wide draws, so they are not
            found significant on the decoder prediction alone.

        Returns
        -------
        Array or DataFrame of shape ``(cells, junctions)`` with DM-normalized PSI*.
        """
        # Validate and prepare
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        # Select cells
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, size=n_samples_overall)

        # Data loader
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        # Batches to transform over
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        # Junction mask and column names
        all_junc = adata.var_names[self.n_genes : self.n_genes + self.n_junctions]
        if junction_list is None:
            junction_mask = slice(None)
            out_cols = list(all_junc)
        else:
            keep = [j in junction_list for j in all_junc]
            junction_mask = keep
            out_cols = [j for j, k in zip(all_junc, keep) if k]

        eps = 1e-8
        results = []

        for tensors in scdl:
            # Precompute per-junction concentration c_j
            device = next(self.module.parameters()).device
            raw_phi = self.module.get_phi()

            if self.dm_concentration == "atse":
                j2a = self.module.junc2atse
                if j2a.device != device:
                    j2a = j2a.to(device)
                if raw_phi.dim() != 1:
                    raise RuntimeError("Expected per-ATSE phi as 1-D when dm_concentration='atse'.")
                # (J,G) @ (G,1) -> (J,)
                phi_j = torch.sparse.mm(j2a, raw_phi.unsqueeze(1)).squeeze(1)

                # subset phi_j to match junction_mask
                if isinstance(junction_mask, list):
                    if len(junction_mask) != phi_j.shape[0]:
                        raise RuntimeError(
                            f"junction_mask length ({len(junction_mask)}) does not match phi_j "
                            f"length ({phi_j.shape[0]})."
                        )
                    phi_j = phi_j[junction_mask]
            else:
                # scalar phi
                phi_j = raw_phi.reshape(1)


            # Pull counts for this minibatch
            y = tensors.get("junc_counts_key", None)
            n = tensors.get("atse_counts_key", None)
            if y is None or n is None:
                raise RuntimeError("Both 'junc_counts_key' and 'atse_counts_key' must be present for DM-normalized PSI.")
            y = y.to(device).float()
            n = n.to(device).float()

            # Optional junction subsetting done after decoding; counts must match mask too
            if isinstance(junction_mask, list):
                y = y[:, junction_mask]
                n = n[:, junction_mask]

            per_batch_psis = []
            for batch in track(transform_batch, disable=silent):
                if batch is not None:
                    b_idx = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(b_idx) * batch

                # Decode p exactly like the original method
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                p = generative_outputs["p"]  # shape: (S?, B, J) or (B, J)
                # Apply junction selection to decoded p
                if isinstance(junction_mask, list):
                    p = p[..., junction_mask]

                # Build phi broadcast to p’s shape
                if p.dim() == 3:
                    # (S, B, J)
                    if phi_j.dim() == 1:
                        phi = phi_j.unsqueeze(0).unsqueeze(0)  # (1,1,J)
                    else:
                        phi = phi_j  # scalar → broadcast
                    y_b = y.unsqueeze(0)  # (1, B, J)
                    n_b = n.unsqueeze(0)  # (1, B, J)
                else:
                    # (B, J)
                    if phi_j.dim() == 1:
                        phi = phi_j.unsqueeze(0)  # (1, J)
                    else:
                        phi = phi_j  # scalar → broadcast
                    y_b = y
                    n_b = n

                # Compute DM-smoothed PSI*
                if sample:
                    a_ = (phi * p + y_b).clamp_min(eps)
                    b_ = (phi * (1.0 - p) + (n_b - y_b)).clamp_min(eps)
                    psi_star = torch.distributions.Beta(a_, b_).sample()
                else:
                    psi_star = (phi * p + y_b) / (phi + n_b + eps)
                psi_star = psi_star.clamp(eps, 1.0 - eps)

                per_batch_psis.append(psi_star)

            # average over transform_batch list
            psi_mb = torch.stack(per_batch_psis, dim=0).mean(0)  # keep sample dim if present
            results.append(psi_mb)

        # Concatenate minibatches along cells
        psis = torch.cat(results, dim=-2)  # align with original method behavior

        # If multiple samples and return_mean, average over samples
        if psis.dim() == 3 and return_mean:
            psis = psis.mean(0)  # (B, J)

        psis_np = psis.detach().cpu().numpy()

        if return_numpy:
            return psis_np
        else:
            return pd.DataFrame(
                psis_np,
                index=adata.obs_names[indices],
                columns=out_cols,
            )


    @torch.inference_mode()
    def get_normalized_expression_ZINB(
        self,
        adata: AnnOrMuData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
        silent: bool = True,
        sample: bool = False,
        return_counts: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns **ZINB posterior-mean** normalized expression (the expression analogue of
        :meth:`get_normalized_splicing_DM`). Only valid for ``gene_likelihood="zinb"``.

        The ZINB is a Gamma-Poisson mixed with a point mass at zero. With decoder mean :math:`\mu`
        (``px_rate``), inverse dispersion :math:`\theta` (``px_r``) and dropout probability
        :math:`\pi = \sigma(\text{px\_dropout})`, the posterior mean of the underlying rate given the
        observed count :math:`y` is

        * :math:`y > 0`: :math:`E[\lambda \mid y] = \mu (\theta + y) / (\theta + \mu)`, i.e. a weighted average
          :math:`\tfrac{\theta}{\theta+\mu}\mu + \tfrac{\mu}{\theta+\mu} y` of the decoder and the data;
        * :math:`y = 0`: with :math:`w = P(\text{dropout} \mid y=0) = \pi / (\pi + (1-\pi)(\theta/(\theta+\mu))^\theta)`,
          :math:`E[\lambda \mid 0] = w \mu + (1-w)\, \theta\mu / (\theta+\mu)` (a dropout zero carries no
          information; a sampling zero pulls the estimate down).

        The result is divided by the cell's library factor (``px_rate / px_scale``), so it is on the same
        scale as :meth:`get_normalized_expression` (``px_scale``) and can replace it in DE. Parameters
        and return shape are the same as :meth:`get_normalized_expression`, plus:

        sample
            If True, return one random draw from the posterior instead of its mean: for y > 0,
            :math:`\lambda \sim \mathrm{Gamma}(\theta + y,\ \theta/\mu + 1)`; for y = 0, with probability w the prior
            :math:`\mathrm{Gamma}(\theta, \theta/\mu)` (dropout zero) and otherwise :math:`\mathrm{Gamma}(\theta, \theta/\mu + 1)`
            (Gamma shape / rate). Used by DE with ``norm_expression_function="zinb_posterior_sample"``, so each Monte
            Carlo draw carries the uncertainty of each cell's expression given its counts.
        return_counts
            If True, do not divide by the library factor (values on the count scale, e.g. for pseudobulk sums).
        """
        if getattr(self.module, "gene_likelihood", None) != "zinb":
            raise ValueError(
                "get_normalized_expression_ZINB requires gene_likelihood='zinb' "
                f"(this model uses {getattr(self.module, 'gene_likelihood', None)!r})."
            )
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)
        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names[: self.n_genes]
            gene_mask = [gene in gene_list for gene in all_genes]

        eps = 1e-8
        exprs = []
        for tensors in scdl:
            y = tensors[REGISTRY_KEYS.X_KEY][:, : self.n_genes]
            per_batch_exprs = []
            for batch in track(transform_batch, disable=silent):
                if batch is not None:
                    batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(batch_indices) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                mu = generative_outputs["px_rate"]
                scale = generative_outputs["px_scale"]
                theta = generative_outputs["px_r"]
                pi = torch.sigmoid(generative_outputs["px_dropout"])
                yb = y.to(mu.device).float()
                if mu.dim() == 3:
                    yb = yb.unsqueeze(0)
                theta = theta.expand_as(mu) if theta.dim() < mu.dim() or theta.shape != mu.shape else theta
                post_pos = mu * (theta + yb) / (theta + mu + eps)
                post_zero_nb = theta * mu / (theta + mu + eps)
                # P(y=0 | NB) = (theta/(theta+mu))^theta, in log space for stability
                log_p0 = theta * (torch.log(theta + eps) - torch.log(theta + mu + eps))
                w = pi / (pi + (1.0 - pi) * torch.exp(log_p0) + eps)
                post_zero = w * mu + (1.0 - w) * post_zero_nb
                post = torch.where(yb > 0, post_pos, post_zero)
                if sample:
                    rate_post = theta / (mu + eps) + 1.0
                    rate_prior = theta / (mu + eps)
                    g_pos = torch.distributions.Gamma(theta + yb, rate_post).sample()
                    g_zero_drop = torch.distributions.Gamma(theta, rate_prior + eps).sample()
                    g_zero_real = torch.distributions.Gamma(theta, rate_post).sample()
                    is_drop = torch.rand_like(w) < w
                    post = torch.where(yb > 0, g_pos, torch.where(is_drop, g_zero_drop, g_zero_real))
                if return_counts:
                    output = post[..., gene_mask]
                else:
                    # back to the px_scale scale: divide by the library factor px_rate / px_scale
                    lib = mu.sum(-1, keepdim=True) / (scale.sum(-1, keepdim=True) + eps)
                    output = (post / (lib + eps))[..., gene_mask]
                per_batch_exprs.append(output.cpu().numpy())
            exprs += [np.stack(per_batch_exprs).mean(0)]

        if n_samples > 1:
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)
        if return_numpy:
            return exprs
        return pd.DataFrame(
            exprs,
            columns=adata.var_names[: self.n_genes][gene_mask],
            index=adata.obs_names[indices],
        )

    @de_dsp.dedent
    @torch.inference_mode()
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        norm_expression_function: Literal["decoder", "zinb_expected", "zinb_posterior_mean", "zinb_posterior_sample"] = "zinb_expected",
        seed: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Differential expression analysis.

        Performs differential gene expression analysis using normalized gene expression estimates.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        norm_expression_function
            Which expression estimate the two groups are compared on:

            * ``"zinb_expected"`` (default): the ZINB expected expression ``(1 - dropout) * px_scale``. With ZINB the
              dropout probability can absorb part of a difference between groups, which ``px_scale`` alone then
              misses (on Allen Brain data downsampled to EasySci depth this roughly doubled the fold changes towards
              the full-depth ones). For a model with ``gene_likelihood="nb"`` or ``"poisson"`` there is no dropout,
              so this is the decoder output.
            * ``"decoder"``: the decoder output ``px_scale`` (:meth:`get_normalized_expression`), as in scVI and
              MultiVI.
            * ``"zinb_posterior_mean"``: the ZINB posterior mean given each cell's observed counts
              (:meth:`get_normalized_expression_ZINB`), which weighs in the raw data.
            * ``"zinb_posterior_sample"``: one draw from that posterior per Monte Carlo sample, so per-cell count
              uncertainty enters ``proba_de``.

            The two posterior options need ``gene_likelihood="zinb"``.
        seed
            If given, ``scvi.settings.seed`` is set to this value before sampling, so that repeated runs give the
            same table. This seeds PyTorch, NumPy and Python's ``random`` for the whole session. Default ``None``
            leaves the random state as it is.
        **kwargs
            Additional keyword arguments for differential computation.

        Returns
        -------
        A pandas DataFrame with differential expression results, sorted by ``proba_de`` (highest first). Features with
        equal ``proba_de`` keep their order in the data, and the FDR flags use the same order, so the table is the same
        on every machine.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        if seed is not None:
            settings.seed = seed

        col_names = adata.var_names[: self.n_genes]

        # Without zero inflation (NB or Poisson models) the expected expression is the decoder output itself.
        uses_zinb = getattr(self.module, "gene_likelihood", None) == "zinb"
        if norm_expression_function == "zinb_expected" and not uses_zinb:
            norm_expression_function = "decoder"

        if norm_expression_function == "decoder":
            model_fn = partial(
                self.get_normalized_expression,
                batch_size=batch_size,
            )
        elif norm_expression_function in ("zinb_expected", "zinb_posterior_mean", "zinb_posterior_sample"):
            if not uses_zinb:
                raise ValueError(
                    f"norm_expression_function={norm_expression_function!r} requires gene_likelihood='zinb' "
                    f"(this model uses {getattr(self.module, 'gene_likelihood', None)!r})."
                )
            if norm_expression_function == "zinb_expected":
                model_fn = partial(self.get_normalized_expression, batch_size=batch_size, expected=True)
            else:
                model_fn = partial(
                    self.get_normalized_expression_ZINB,
                    batch_size=batch_size,
                    sample=(norm_expression_function == "zinb_posterior_sample"),
                )
        else:
            raise ValueError(
                "norm_expression_function must be 'decoder', 'zinb_expected', 'zinb_posterior_mean' or "
                "'zinb_posterior_sample', "
                f"got {norm_expression_function!r}"
            )
        all_stats_fn = partial(
            scrna_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[: self.n_genes],
        )
        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )
        return _sort_and_flag_stably(result, col_names, mode, fdr_target)


    @de_dsp.dedent
    @torch.inference_mode()
    def differential_splicing(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.10,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        norm_splicing_function: Literal["decoder", "dm_posterior_mean", "dm_posterior_sample"] = "dm_posterior_sample",
        seed: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        r"""Differential splicing analysis.

        Performs differential junction usage analysis using normalized PSI
        from the model's splicing decoder.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        norm_splicing_function
            Which junction-usage estimate the two groups are compared on:

            * ``"dm_posterior_sample"`` (default): one draw per Monte Carlo sample from each cell's Beta posterior of
              junction usage, which combines the decoded usage (weighted by the ATSE's concentration) with the cell's
              own reads (:meth:`get_normalized_splicing_DM` with ``sample=True``), so per-cell read uncertainty enters
              ``proba_ds``.
            * ``"dm_posterior_mean"``: the mean of that posterior (``sample=False``).
            * ``"decoder"``: the decoder's junction usage alone (:meth:`get_normalized_splicing`).

            ``delta`` (default 0.1) is a threshold on the absolute difference in junction usage. The paper also keeps
            only junctions observed in at least 3 cells of each group; apply that with
            :func:`~splicevi.filter_read_support` on the returned table (it does not change which junctions are significant).
        seed
            If given, ``scvi.settings.seed`` is set to this value before sampling, so that repeated runs give the
            same table. This seeds PyTorch, NumPy and Python's ``random`` for the whole session. Default ``None``
            leaves the random state as it is.
        **kwargs
            Additional keyword arguments for differential computation.

        Returns
        -------
        A pandas DataFrame with differential splicing results, one row per junction in the order of the data. Among
        junctions with equal ``proba_ds`` at the FDR cutoff, the ones earlier in the data are significant, so the
        result is the same on every machine.
        """

        def scsplicing_ratio_properties(
            adata_manager: AnnDataManager,
            idx1: list[int] | np.ndarray,
            idx2: list[int] | np.ndarray,
            var_idx: list[int] | np.ndarray | None = None,
        ) -> dict[str, np.ndarray]:
            """Empirical PSI means and effect on the junc_ratio matrix.

            Pulls PSI from "junc_ratio" and returns:
            emp_mean1, emp_mean2, emp_effect = mean2 - mean1
            Observed counts per junction in each group are also returned:
            obs_count1, obs_count2.
            Means are computed only across observed (mask=1) junctions.
            If a junction is not observed in any cell in a group, its mean is set to -1.
            """
            X = adata_manager.get_from_registry("junc_ratio")  # (cells x junctions)
            M = adata_manager.get_from_registry("psi_observed_mask")      # (cells x junctions), 0/1

            X1, X2 = X[idx1], X[idx2]
            M1, M2 = M[idx1], M[idx2]

            if var_idx is not None:
                X1, X2 = X1[:, var_idx], X2[:, var_idx]
                M1, M2 = M1[:, var_idx], M2[:, var_idx]

            def _to_ndarray(A):
                # Works for dense, sparse, and numpy.matrix
                if hasattr(A, "toarray"):
                    return A.toarray()
                return np.asarray(A)

            def masked_mean_and_counts(X_sub, M_sub):
                """Compute mean across observed entries only, plus observed counts."""
                X_sub = _to_ndarray(X_sub)
                M_sub = _to_ndarray(M_sub)

                # Ensure numeric mask (0/1) and float data
                M_sub = M_sub.astype(np.float32, copy=False)
                X_sub = X_sub.astype(np.float32, copy=False)

                # Elementwise multiply; sums over cells -> per-junction vectors
                observed_sum = (X_sub * M_sub).sum(axis=0)
                observed_count = M_sub.sum(axis=0)

                # Safe division with fallback -1 where no observations
                mean = np.divide(
                    observed_sum,
                    observed_count,
                    out=np.full_like(observed_sum, -1.0, dtype=np.float32),
                    where=observed_count > 0,
                )
                # Return 1D vectors
                return mean.ravel(), observed_count.ravel()

            mean1, obs_count1 = masked_mean_and_counts(X1, M1)
            mean2, obs_count2 = masked_mean_and_counts(X2, M2)

            return {
                "emp_mean1": mean1,
                "emp_mean2": mean2,
                "emp_effect": mean2 - mean1,
                "obs_count1": obs_count1,  # number of cells that observed each junction in idx1
                "obs_count2": obs_count2,  # number of cells that observed each junction in idx2
            }

        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        if seed is not None:
            settings.seed = seed

        col_names = adata["splicing"].var_names

        # Junction-usage estimate the groups are compared on.
        if norm_splicing_function == "decoder":
            model_fn = partial(
                self.get_normalized_splicing,
                batch_size=batch_size,
            )
        elif norm_splicing_function in ("dm_posterior_mean", "dm_posterior_sample"):
            model_fn = partial(
                self.get_normalized_splicing_DM,
                batch_size=batch_size,
                sample=(norm_splicing_function == "dm_posterior_sample"),
            )
        else:
            raise ValueError(
                "norm_splicing_function must be 'decoder', 'dm_posterior_mean' or 'dm_posterior_sample', "
                f"got {norm_splicing_function!r}"
            )

        # Empirical statistics of the observed PSI
        all_stats_fn = partial(
            scsplicing_ratio_properties,
            var_idx=np.arange(adata["splicing"].shape[1])[: self.n_junctions],
        )

        # PSI is a bounded [0, 1] proportion reported downstream as a raw difference
        # (`effect_size`); use that same scale for the `delta` hypothesis test instead
        # of scvi-tools' default log2 fold-change, so `delta` means "minimum |ΔPSI|".
        kwargs.setdefault("change_fn", _psi_absolute_change_fn)

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            pseudocounts=1e-6,
            **kwargs,
        )
        result = _sort_and_flag_stably(result, col_names, mode, fdr_target)
        result = pd.DataFrame(
            {
                "proba_ds": result.proba_de,
                "is_ds_fdr": result.loc[:, f"is_de_fdr_{fdr_target}"],
                "bayes_factor": result.bayes_factor,
                "effect_size": result.scale2 - result.scale1,
                "emp_effect": result.emp_mean2 - result.emp_mean1,
                "est_prob1": result.scale1,
                "est_prob2": result.scale2,
                "emp_prob1": result.emp_mean1,
                "emp_prob2": result.emp_mean2,
                "n_obs_group1": (
                    result["obs_count1"].astype(int)
                    if "obs_count1" in result.columns else None
                ),
                "n_obs_group2": (
                    result["obs_count2"].astype(int)
                    if "obs_count2" in result.columns else None
                ),
            },
            index=col_names,
        )

        # Readable junction and gene names first, when the data has them.
        var_cols = adata["splicing"].var
        wanted = [c for c in ["junction_id", "gene_name"] if c in var_cols.columns]
        if wanted:
            meta = var_cols.loc[col_names, wanted]  # reindex to the same junctions in the same order
            result = result.join(meta)

        if "junction_id" in result.columns:
            result.insert(0, "junction_id", result.pop("junction_id"))
        if "gene_name" in result.columns:
            result.insert(1, "gene_name", result.pop("gene_name"))

        return result

    # --- Decoder covariates and decoder fine-tuning (transfer to a new atlas) ---

    def add_decoder_covariates(self, n_covariates: int = 1) -> None:
        """Add zero-initialized covariate inputs to the first hidden layer of both decoders.

        The model's output is unchanged until the covariate weights are trained (:meth:`finetune_decoders`). Recorded in
        the model's init parameters (``n_decoder_covariates``), so :meth:`save` / :meth:`load` keep the inputs.

        Parameters
        ----------
        n_covariates
            Number of covariate inputs (1 for technology: reference 0, query 1).
        """
        self.module.add_decoder_covariates(n_covariates)
        params = self.init_params_.get("non_kwargs", self.init_params_)
        params["n_decoder_covariates"] = int(n_covariates)

    def set_decoder_covariate(self, value) -> None:
        """Value of the decoder covariate input(s) used for every cell from now on (saved with the model).

        After :meth:`finetune_decoders` the model decodes as query data (``value = 1.0``); set ``0.0`` to decode as the
        reference data. Affects every decoder output: normalized expression / splicing, differential expression and
        differential splicing.

        Parameters
        ----------
        value
            A number (all covariates) or one value per covariate.
        """
        k = self.module.n_decoder_covariates
        if not k:
            raise ValueError("this model has no decoder covariate inputs (see finetune_decoders / add_decoder_covariates)")
        value = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
        if value.numel() not in (1, k):
            raise ValueError(f"expected 1 or {k} decoder covariate value(s), got {value.numel()}")
        with torch.no_grad():
            self.module.decoder_covariate_value.copy_(value.expand(k))

    @contextmanager
    def decoder_covariate(self, value):
        """Context manager: decode with this decoder covariate value inside the block, then restore the previous one.

        Example
        -------
        >>> with model.decoder_covariate(0.0):   # decode the query cells as if they were reference data
        ...     psi_as_reference = model.get_normalized_splicing_DM(adata=query_mdata)
        """
        previous = self.module.decoder_covariate_value.detach().clone()
        self.set_decoder_covariate(value)
        try:
            yield self
        finally:
            with torch.no_grad():
                self.module.decoder_covariate_value.copy_(previous)

    def finetune_decoders(
        self,
        query_mdata: MuData,
        reference_mdata: MuData,
        n_query_per_step: int = 32,
        n_reference_per_step: int = 128,
        lr: float = 1e-4,
        max_epochs: int = 300,
        early_stopping_patience: int = 30,
        validation_fraction: float = 0.1,
        seed: int = 20261137,
        set_torch_seed: bool = True,
        max_minutes: float | None = None,
        checkpoint_dir: str | None = None,
        fix_depth_inputs: bool = True,
        query_covariate_value: float = 1.0,
        reference_covariate_value: float = 0.0,
    ) -> pd.DataFrame:
        """Adapt a trained model to a new atlas by fine-tuning only its decoders, plus one technology input.

        Use when this model was trained on a reference atlas (e.g. Allen Brain, deep Smart-seq) and you want its
        expression and splicing estimates for a new, sparser atlas of the same tissue (e.g. EasySci, single-nucleus 3')
        without retraining from scratch. The encoders are not changed, so query cells are placed in the reference
        latent space; the decoders learn the query data's expression and splicing levels. Defaults are the same as what's used in 
        the paper's EasySci model.

        What happens:

        1. One zero-initialized decoder covariate input is added to the first hidden layer of both decoders
           (technology: ``reference_covariate_value`` for reference cells, ``query_covariate_value`` for query cells).
           At zero weights the model is the reference model.
        2. Everything is frozen except the two decoders and the covariate weights. Kept from the reference model: the
           encoders, junction embeddings and latent mixing, the deviation input's population PSI and pool divisor,
           the per-ATSE splicing concentrations and the gene dispersions (re-learning the last two on the query data
           predicted held-out query reads worse in the paper's tests, so it is not offered).
        3. With ``fix_depth_inputs`` (models with ``decoder_depth_covariates``), the decoders' depth inputs are fixed
           at the reference training mean for every cell, since the reference depth statistics do not describe the
           query data. The cross-modality gate is set open, as at the end of training.
        4. Each step uses ``n_query_per_step`` query cells and ``n_reference_per_step`` reference cells (the reference
           "replay" keeps the decoders fitting the reference); loss = query loss + reference loss; Adam at ``lr``.
        5. A random ``validation_fraction`` of the query cells is held out for early stopping (patience
           ``early_stopping_patience`` epochs, at most ``max_epochs``, optional wall-clock limit ``max_minutes``).
        6. The best state is restored and the model is left decoding as query data (decoder covariate =
           ``query_covariate_value``), ready for :meth:`differential_splicing` etc. on the query cells.

        Calling it again continues from the current weights with a fresh optimizer. Training a fine-tuned model with
        :meth:`train` is not supported: every cell would get ``decoder_covariate_value``, so the covariate weights
        would act as a plain bias.

        Parameters
        ----------
        query_mdata
            The new atlas (e.g. EasySci training metacells) in this model's gene and junction space (same genes, same
            junction ids and ATSE assignment), with the layers this model was trained on: expression counts
            (``length_norm``; raw counts for UMI data), library size (``X_library_size``), junction and ATSE counts,
            ``junc_ratio`` and ``psi_mask``. If not yet set up, it is set up with this model's setup arguments (as
            :meth:`load` does). The paper's reproduction code (SpliceVI-utils) shows how these files were built from the
            model-ready data.
        reference_mdata
            A replay set of the data this model was trained on (e.g. 20,000 Allen training cells), same layers.
        n_query_per_step, n_reference_per_step
            Query and reference cells per optimization step (paper: 32 and 128).
        lr
            Adam learning rate.
        max_epochs, early_stopping_patience
            Epochs over the query training cells; stop when the validation loss has not improved for this many epochs.
        validation_fraction
            Share of query cells held out (never trained on) to decide when to stop.
        seed
            Seed of the validation split, the minibatch order and the reference cells per step (numpy), and, with
            ``set_torch_seed``, of torch's global random state (latent sampling and decoder dropout during
            fine-tuning). GPU arithmetic is not bit-deterministic, so reruns agree closely but not exactly.
        set_torch_seed
            Call ``torch.manual_seed(seed)`` (changes torch's global random state). False when the caller seeded torch
            already, e.g. before loading the model, as the paper's fine-tuning run did.
        max_minutes
            Optional wall-clock limit (the best state so far is kept). Keep it below a job's time limit.
        checkpoint_dir
            If set, the best module state is also written to ``<checkpoint_dir>/module_state_best.pt`` at every
            improvement, so a run killed from outside (e.g. a job time limit) keeps its best state.
        fix_depth_inputs
            Fix the decoders' depth inputs at the reference training mean (models with decoder depth covariates).
        query_covariate_value, reference_covariate_value
            Values of the technology input for query and reference cells.

        Returns
        -------
        Training history: one row per epoch (``epoch``, ``train_loss``, ``validation_loss``; epoch -1 = before
        training). Also stored as ``finetune_history_``, with the settings and outcome in ``finetune_params_``; both
        are saved by :meth:`save`.

        Examples
        --------
        >>> model = SPLICEVI.load(reference_model_dir, adata=query_mdata)
        >>> model.to_device("cuda")
        >>> history = model.finetune_decoders(query_mdata, reference_mdata)
        >>> model.save(out_dir)                         # SPLICEVI.load(out_dir, adata=query_mdata) decodes as query
        >>> ds = model.differential_splicing(adata=query_mdata, groupby="broad_cell_type", group1="A", group2="B")
        """
        from .finetune import finetune_decoders as _finetune_decoders
        return _finetune_decoders(self, query_mdata, reference_mdata, n_query_per_step=n_query_per_step,
                                  n_reference_per_step=n_reference_per_step, lr=lr, max_epochs=max_epochs,
                                  early_stopping_patience=early_stopping_patience,
                                  validation_fraction=validation_fraction, seed=seed, set_torch_seed=set_torch_seed,
                                  max_minutes=max_minutes, checkpoint_dir=checkpoint_dir,
                                  fix_depth_inputs=fix_depth_inputs, query_covariate_value=query_covariate_value,
                                  reference_covariate_value=reference_covariate_value)

    @staticmethod
    def preset(name, atlas=None, joint=False):
        """Settings of the paper's models (the class defaults equal ``preset("paper")``); see
        :func:`splicevi.presets.preset`."""
        from .presets import preset as _preset
        return _preset(name, atlas=atlas, joint=joint)

    @classmethod
    def setup_anndata(cls, adata: AnnData, **kwargs):
        """Not supported: SPLICEVI needs a MuData object with an ``"rna"`` and a ``"splicing"`` modality.

        Use :meth:`setup_mudata`. (scvi-tools requires every model class to define this method.)
        """
        raise NotImplementedError(
            "SPLICEVI works on MuData objects only. Put expression and splicing in a MuData with modalities 'rna' "
            "and 'splicing' and call SPLICEVI.setup_mudata(mdata)."
        )

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: str | None = "length_norm",
        junc_ratio_layer: str | None = "junc_ratio",
        atse_counts_layer: str | None = "cell_by_cluster_matrix",
        junc_counts_layer: str | None = "cell_by_junction_matrix",
        psi_mask_layer: str | None = "psi_mask",
        batch_key: str | None = None,
        size_factor_key: str | None = "X_library_size",
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        idx_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        **kwargs,
    ):
        r"""Set up a MuData object for SPLICEVI.

        The defaults are the layer and modality names of the paper's MuData files, so a file in that layout needs only
        ``SPLICEVI.setup_mudata(mdata)`` (plus ``batch_key`` if there is one).

        Parameters
        ----------
        %(param_mdata)s
        rna_layer
            Layer of the RNA modality with the gene expression counts. Default ``"length_norm"``.
            If `None`, the primary data (`.X`) of that modality is used.
        junc_ratio_layer
            Layer of the splicing modality with the junction usage (PSI) per cell and junction. Default
            ``"junc_ratio"``. If `None`, the primary data (`.X`) of that modality is used.
        atse_counts_layer
            Layer of the splicing modality with the total reads of each junction's ATSE (repeated for every junction of
            the ATSE). Default ``"cell_by_cluster_matrix"``.
        junc_counts_layer
            Layer of the splicing modality with the reads of each junction. Default ``"cell_by_junction_matrix"``.
        psi_mask_layer
            Layer of the splicing modality with the observation mask (1 = observed, 0 = missing) per cell and junction.
            Default ``"psi_mask"``.
        %(param_batch_key)s
        size_factor_key
            Key in ``mdata.obsm`` for the library size factors. Default ``"X_library_size"``.
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(idx_layer)s
        modalities
            Which modality holds each kind of data. Default ``{"rna_layer": "rna", "junc_ratio_layer": "splicing"}``:
            expression in ``mdata["rna"]`` and every splicing layer in ``mdata["splicing"]``.

        Examples
        --------
        >>> mdata = mu.MuData({"rna": expression_anndata, "splicing": splicing_anndata})
        >>> SPLICEVI.setup_mudata(mdata)                          # the paper's layer names
        >>> SPLICEVI.setup_mudata(mdata, rna_layer="raw_counts")  # expression counts in another layer
        >>> model = SPLICEVI(mdata)
        """
        # Default layout: expression in mdata["rna"], every splicing layer in mdata["splicing"] (filled in before the
        # setup arguments are recorded, so a saved model stores the mapping it used).
        if modalities is None:
            modalities = {"rna_layer": "rna", "junc_ratio_layer": "splicing"}
        setup_method_args = cls._get_setup_method_args(**locals())
        modalities = cls._create_modalities_attr_dict(modalities, setup_method_args)
        mdata.obs["_indices"] = np.arange(mdata.n_obs)

        batch_field = fields.MuDataCategoricalObsField(
            REGISTRY_KEYS.BATCH_KEY,
            batch_key,
            mod_key=modalities.batch_key,
        )

        if size_factor_key is None:
            size_factor_key = "X_library_size"

        mudata_fields = [
            batch_field,
            fields.MuDataCategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None, mod_key=None),
            fields.MuDataObsmField(REGISTRY_KEYS.SIZE_FACTOR_KEY, attr_key=size_factor_key,is_count_data=False),
            fields.MuDataCategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys, mod_key=modalities.categorical_covariate_keys),
            fields.MuDataNumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys, mod_key=modalities.continuous_covariate_keys),
            fields.MuDataNumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices", mod_key=modalities.idx_layer, required=False),
        ]

        # RNA modality registration: use rna_layer from the GE AnnData.
        if modalities.rna_layer is not None:
            mudata_fields.append(
                fields.MuDataLayerField(
                    REGISTRY_KEYS.X_KEY,
                    rna_layer,  # e.g. "raw_counts"
                    mod_key=modalities.rna_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )

        # Splicing modality registration: we expect the ATSE AnnData to hold the relevant splicing layers.
        if modalities.junc_ratio_layer is not None:
            # Register the primary splicing data as X from the specified junc_ratio_layer.
            mudata_fields.append(
                fields.MuDataLayerField(
                    "junc_ratio",
                    junc_ratio_layer,  # e.g. "junc_ratio"
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
            # Register the additional splicing layers.
            if atse_counts_layer is None:
                atse_counts_layer = "cell_by_cluster_matrix"
            mudata_fields.append(
                fields.MuDataLayerField(
                    "atse_counts_key",  # internal key used by the model
                    atse_counts_layer,
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )
            if junc_counts_layer is None:
                junc_counts_layer = "cell_by_junction_matrix"
            mudata_fields.append(
                fields.MuDataLayerField(
                    "junc_counts_key",  # internal key used by the model
                    junc_counts_layer,
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=True,
                    mod_required=True,
                )
            )

            if psi_mask_layer is None:
                psi_mask_layer = "psi_mask"
            mudata_fields.append(
                fields.MuDataLayerField(
                    "psi_observed_mask",  # internal key used by the model
                    psi_mask_layer,
                    mod_key=modalities.junc_ratio_layer,
                    is_count_data=False,
                    mod_required=True,
                )
            )

        adata_manager = AnnDataManager(fields=mudata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(mdata, **kwargs)
        cls.register_manager(adata_manager)


    def _check_adata_modality_weights(self, adata):
        r"""Checks whether held-out data is provided when using per-cell weights.

        Parameters
        ----------
        adata : AnnData or MuData
            The input data.

        Raises
        ------
        RuntimeError
            If held-out data is provided when per-cell modality weights are used.
        """
        if (adata is not None) and (self.module.modality_weights == "cell"):
            raise RuntimeError("Held out data not permitted when using per cell weights")
