from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from mudata import MuData
from scipy.sparse import csr_matrix, vstack
from torch.distributions import Normal

from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager, fields
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
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


import torch
from scvi.train import AdversarialTrainingPlan

from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

class MyAdvTrainingPlan(AdversarialTrainingPlan):

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        # keep all your existing AdversarialTrainingPlan args here…
        lr_scheduler_type: Literal["plateau", "step"] = "plateau",
        step_size: int = 10,
        gradient_clipping: bool = True,
        gradient_clipping_max_norm: float = 5.0,
        cross_gate_mode: str = "hard",
        **kwargs,
    ):
        super().__init__(module=module, **kwargs)
        self.cross_gate_mode = cross_gate_mode  # "hard" or "soft"
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
        # KL warmup drives self.kl_weight from 0→1
        if self.cross_gate_mode == "soft":
            return float(self.kl_weight)                 # gradually open
        # print(self.kl_weight)
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
        print("Configuring optimizers")
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.lr_scheduler_type == "step":
            print("Using step LR")
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
            print("Using plateau LR")
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
    """Integration of gene expression and alternative splicing signals.

    SPLICEVI integrates gene expression and alternative splicing (junction-usage) modalities,
    learning a joint latent space with reconstruction heads per modality. It wraps the
    :class:`~scvi.module.SPLICEVAE` module and adds training utilities and convenience APIs.

    Parameters
    ----------
    adata
        AnnData/MuData object that has been registered via
        :meth:`~scvi.model.SPLICEVI.setup_anndata` or
        :meth:`~scvi.model.SPLICEVI.setup_mudata`.

    # --- Modality mixing ---
    modality_weights
        How to weight modalities when forming the joint latent:
        * ``"equal"`` – equal weight per modality,
        * ``"universal"`` – one global weight per modality (learned),
        * ``"cell"`` – per-cell weights (learned),
        * ``"concatenate"`` – do not mix; concatenate per-modality latents,
        * ``"per_dimension_weighted_average"`` – one learned splicing weight per latent
          dimension (expression weight = 1 - splicing weight), constrained so the mean
          weight stays at 0.5 to prevent either modality from dominating globally.
    modality_penalty
        Alignment penalty across modalities: ``"Jeffreys"``, ``"MMD"``, or ``"None"``.

    # --- Likelihoods & dispersion (expression) ---
    gene_likelihood
        Expression likelihood: ``"zinb"``, ``"nb"``, or ``"poisson"``.
    dispersion
        Expression dispersion parameterization: ``"gene"``, ``"gene-batch"``, ``"gene-label"``, or ``"gene-cell"``.

    # --- Architecture toggles ---
    splicing_encoder_architecture
        Splicing encoder:
        * ``"vanilla"`` – SCVI `Encoder`,
        * ``"partial"`` – `PartialEncoderEDDIFaster`.
    splicing_decoder_architecture
        Splicing decoder:
        * ``"vanilla"`` – nonlinear `DecoderSplice`,
        * ``"linear"`` – linear decoder.
    expression_architecture
        Expression decoder: ``"vanilla"`` (nonlinear) or ``"linear"`` (linear decoder).

    # --- Shared SCVI-style encoder/decoder hyperparameters ---
    n_hidden
        Width of SCVI encoders/decoders. If `None`, uses √(n_junctions) (capped at 128).
    n_latent
        Joint latent dimensionality (per-modality before concatenation). If `None`, uses √(n_hidden).
    n_layers_encoder
        Hidden layers in encoders (including the post pooling MLP in the partial splicing encoder).
    n_layers_decoder
        Hidden layers in decoders (not used by the linear decoders).
    dropout_rate
        Dropout rate in MLPs.
    use_batch_norm
        Where to apply batch norm: ``"encoder"``, ``"decoder"``, ``"both"``, or ``"none"``.
    use_layer_norm
        Where to apply layer norm: ``"encoder"``, ``"decoder"``, ``"both"``, or ``"none"``.
    latent_distribution
        Latent distribution: ``"normal"`` or ``"ln"`` (logistic normal).
    deeply_inject_covariates
        If True, injects covariates at all decoder layers.
    encode_covariates
        If True, provides covariates to encoders.

    # --- Splicing likelihood ---
    splicing_loss_type
        Splicing reconstruction loss: ``"binomial"``, ``"beta_binomial"``, or ``"dirichlet_multinomial"``.
    splicing_concentration
        Optional concentration for beta binomial. Ignored for binomial.
    dm_concentration
        For Dirichlet multinomial: ``"atse"`` (per ATSE concentration) or ``"scalar"`` (single shared).
    splicing_loss_weight
        Scalar multiplier applied to the splicing reconstruction loss relative to the
        expression reconstruction loss. Default 1.0 (unweighted).

    # --- PartialEncoder knobs (used when splicing_encoder_architecture="partial") ---
    encoder_hidden_dim
        Hidden width inside the PartialEncoder MLP that maps pooled junction codes to latent statistics.
    code_dim
        Dimensionality of per junction embeddings in PartialEncoderEDDIFaster.
    h_hidden_dim
        Hidden width of the shared h network that processes (psi, feature_embedding) for each observed junction.
    pool_mode
        Pooling across observed junctions per cell: ``"mean"`` or ``"sum"``.
    max_nobs
        Optional cap on the number of observed (cell, junction) pairs processed in a single scatter chunk.
        Set to a negative value to disable chunking.

    # --- Dataset shape/bookkeeping (inferred when possible) ---
    n_genes
        Number of gene expression features. Required if `adata` is AnnData.
    n_junctions
        Number of splicing features. Required if `adata` is AnnData.

    # --- Model-only options ---
    initialize_embeddings_from_pca
        If True and using the partial splicing encoder, initialize the per junction embedding
        table via PCA on junction ratios.
    fully_paired
        If True, the model exposes only a joint latent (no modality specific latents).

    **model_kwargs
        Forwarded to :class:`~scvi.module.SPLICEVAE`.
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
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",

        # --- Shared SCVI-style encoder/decoder hyperparameters ---
        n_hidden: int | None = None,
        n_latent: int | None = None,
        n_layers_encoder: int = 2,
        n_layers_decoder: int = 2,
        dropout_rate: float = 0.1,
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
        splicing_concentration: float | None = None,
        splicing_loss_weight: float = 1.0,

        # --- Architecture toggles ---
        splicing_encoder_architecture: Literal["vanilla", "partial"] = "partial",
        splicing_decoder_architecture: Literal["vanilla", "linear"] = "linear",
        expression_architecture: Literal["vanilla", "linear"] = "linear",

        # --- PartialEncoder knobs (splicing_architecture="partial") ---
        encoder_hidden_dim: int = 128,
        code_dim: int = 16,
        h_hidden_dim: int = 64,
        pool_mode: Literal["mean", "sum"] = "mean",
        max_nobs: int = -1,

        # --- Model-only helpers ---
        initialize_embeddings_from_pca: bool = False,
        fully_paired: bool = False,

        # (module bookkeeping inferred from data managers)
        **model_kwargs,
    ):
        super().__init__(adata)

        if n_genes is None or n_junctions is None:
            assert isinstance(adata, MuData), (
                "n_genes and n_junctions must be provided if using AnnData"
            )
            n_genes = self.summary_stats.get("n_vars", 0)
            n_junctions = self.summary_stats.get("n_junc", 0)

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
            splicing_concentration=splicing_concentration,
            dm_concentration=dm_concentration,
            splicing_loss_weight=splicing_loss_weight,

            # architectures
            splicing_encoder_architecture=splicing_encoder_architecture,
            splicing_decoder_architecture=splicing_decoder_architecture,
            expression_architecture=expression_architecture,

            # partial encoder knobs
            encoder_hidden_dim=encoder_hidden_dim,
            code_dim=code_dim,
            h_hidden_dim=h_hidden_dim,
            pool_mode=pool_mode,
            max_nobs=max_nobs,

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
            f"sp_conc={splicing_concentration}, sp_loss_weight={splicing_loss_weight} | "
            f"mix={modality_weights}, penalty={modality_penalty} | "
            f"PE(code_dim={code_dim}, h_hidden={h_hidden_dim}, "
            f"enc_hidden={encoder_hidden_dim}, pool={pool_mode}, "
            f"max_nobs={max_nobs}) | "
            f"init_from_pca={initialize_embeddings_from_pca}"
        )

        self.fully_paired = fully_paired
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        self.n_genes = n_genes
        self.n_junctions = n_junctions
        self.get_normalized_function_name = "get_normalized_splicing"
        self.dm_concentration = dm_concentration

        if self.adata is not None:
            if initialize_embeddings_from_pca and splicing_encoder_architecture == "partial":
                self.init_feature_embedding_from_adata()

            # Only needed for DM splicing likelihood
            if splicing_loss_type == "dirichlet_multinomial":
                self.init_junc2atse()


        
    def make_junc2atse(self, atse_labels):
        print("Making Junc2Atse...")
        num_junctions = len(atse_labels)
        atse_labels = atse_labels.astype('category')
        row_indices = torch.arange(num_junctions, dtype=torch.long)
        col_indices = torch.tensor(atse_labels.cat.codes.values)
        num_atses = len(atse_labels.cat.categories)
        print(f"Found {num_junctions} junctions and {num_atses} ATSEs.")
        from torch import nn
        if self.dm_concentration== "atse":
            print("Using ATSE level concentration, overwriting concentration parameter.")
            self.module.log_phi_j = nn.Parameter(torch.randn(num_atses) * 0.5 + np.log(100.0))
            self.module.log_phi_j.requires_grad_(True)

        return torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=torch.ones(len(row_indices), dtype=torch.float32),
            size=(num_junctions, len(atse_labels.cat.categories))
        ).coalesce()
    
    def init_junc2atse(self) -> None:
        # Return early if already set
        if hasattr(self.module, "junc2atse"):
            return

        cl_info = self.adata_manager.data_registry["atse_counts_key"]
        cl_key, mod_key = cl_info.attr_key, cl_info.mod_key
        cluster_counts = self.adata[mod_key].layers[cl_key]  # existence check
        atse_labels = self.adata[mod_key].var["event_id"]

        j2a = self.make_junc2atse(atse_labels)
        self.module.junc2atse = j2a.coalesce().to(self.module.device)


    def init_feature_embedding_from_adata(self) -> None:
        print("Initializing feature embeddings...")
        jr_info = self.adata_manager.data_registry["junc_ratio"]
        jr_key, mod_key = jr_info.attr_key, jr_info.mod_key
        X = self.adata[mod_key].layers[jr_key]
        # keep X sparse:
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        # fit a (code_dim)-component SVD on the sparse X
        svd = TruncatedSVD(n_components=self.module.z_encoder_splicing.code_dim,
                        algorithm="randomized")
        svd.fit(X)

        # components_.shape == (n_components, n_vars)
        comps = svd.components_.T  # (n_vars, code_dim)

        with torch.no_grad():
            emb = self.module.z_encoder_splicing.feature_embedding
            emb.copy_(torch.as_tensor(comps, dtype=emb.dtype, device=emb.device))
        print("Initialized feature embeddings...")



    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float | None = None,
        validation_size: float | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        early_stopping_patience: int = 50,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 50,
        adversarial_mixing: bool = True,
        lr_scheduler_type: Literal["plateau", "step"] = "plateau",
        reduce_lr_on_plateau: bool = False,
        lr_factor: float = 0.6,
        lr_patience: int = 30,
        lr_threshold: float = 0.0,
        lr_scheduler_metric: Literal[
            "elbo_validation", "reconstruction_loss_validation", "kl_local_validation"
        ] = "elbo_validation",
        step_size: int = 10,
        gradient_clipping: bool = True,
        gradient_clipping_max_norm: float = 5.0,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        r"""Trains the model using amortized variational inference on gene expression and splicing modalities.

        Parameters
        ----------
        max_epochs
            Number of epochs to train over.
        lr
            Learning rate for optimization.
        accelerator, devices
            Hardware acceleration options.
        train_size, validation_size
            Proportions for splitting the data into train and validation sets.
        shuffle_set_split
            Whether to shuffle before splitting.
        batch_size
            Minibatch size for training.
        weight_decay, eps
            Optimizer hyperparameters.
        early_stopping
            Whether to enable early stopping.
        early_stopping_patience
            Number of epochs with no improvement before stopping.
        check_val_every_n_epoch
            How often (in epochs) to run validation.
        n_steps_kl_warmup, n_epochs_kl_warmup
            KL divergence warmup parameters (by steps or epochs).
        adversarial_mixing
            Whether to include adversarial classifier during training.
        lr_scheduler_type
            Scheduler type in TrainingPlan: “plateau” (reduce-on-plateau) or “step” (fixed-step).
        reduce_lr_on_plateau
            If True and using plateau scheduler, enable ReduceLROnPlateau.
        lr_factor
            Multiplicative factor for LR reduction (used for both plateau and step schedulers).
        lr_patience
            Number of epochs with no improvement for plateau scheduler.
        lr_threshold
            Threshold for measuring new optimum (plateau scheduler).
        lr_scheduler_metric
            Metric to monitor for plateau scheduler.
        step_size
            Epoch interval between LR drops (step scheduler).
        gradient_clipping
            Whether or not (true or false) to use gradient norm clipping
        gradient_clipping_max_norm
            Max norm of the gradients to be used in gradient clipping
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
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = datasplitter_kwargs or {}

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
        %(de_silent)s

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is
        :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
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
            raw_phi = torch.nn.functional.softplus(self.module.log_phi_j)

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
        **kwargs
            Additional keyword arguments for differential computation.

        Returns
        -------
        A pandas DataFrame with differential expression results.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)

        col_names = adata.var_names[: self.n_genes]
        model_fn = partial(
            self.get_normalized_expression,
            batch_size=batch_size,
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
        return result


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
        norm_splicing_function: Literal["decoder", "dm_posterior_mean"] = "dm_posterior_mean",
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
        **kwargs
            Additional keyword arguments for differential computation.

        Returns
        -------
        A pandas DataFrame with differential splicing results.
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

        col_names = adata["splicing"].var_names


        # Use expected PSI from the splicing head

        if norm_splicing_function == "decoder":
            model_fn = partial(
                self.get_normalized_splicing,
                batch_size=batch_size,
            )
        elif norm_splicing_function == "dm_posterior_mean":
            model_fn = partial(
                self.get_normalized_splicing_DM,
                batch_size=batch_size,
            )
        else:
            print("Unrecognized Normalized Splicing Mode, defaulting to regular decoder output.")
            model_fn = partial(
                self.get_normalized_splicing,
                batch_size=batch_size,
            )


        # Minimal empirical stats on PSI itself
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


    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        junc_ratio: str | None = None,
        cell_by_junction_matrix: str | None = None,
        cell_by_cluster_matrix: str | None = None,
        psi_mask_layer: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        **kwargs,
    ):
        r"""Set up an AnnData object for SPLICEVI.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        junc_ratio
            Key in ``adata.layers`` for junction ratio values.
        cell_by_junction_matrix
            Key in ``adata.layers`` for the cell-by-junction matrix.
        cell_by_cluster_matrix
            Key in ``adata.layers`` for the cell-by-cluster splicing matrix.
        psi_mask_layer
            Layer with binary mask (1=observed, 0=missing) per junction.
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s

        Notes
        -----
        Use this method if your splicing data is stored in an AnnData object where gene expression and
        splicing features are concatenated.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            batch_field,
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, None),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            NumericalJointObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if junc_ratio is not None:
            anndata_fields.append(LayerField("junc_ratio_key", junc_ratio, is_count_data=True))
        if cell_by_junction_matrix is not None:
            anndata_fields.append(LayerField("cell_by_junction_matrix", cell_by_junction_matrix, is_count_data=True))
        if cell_by_cluster_matrix is not None:
            anndata_fields.append(LayerField("cell_by_cluster_matrix", cell_by_cluster_matrix, is_count_data=True))
        if psi_mask_layer is not None:
            anndata_fields.append(LayerField("psi_observed_mask", psi_mask_layer, is_count_data=False))
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_mudata(
        cls,
        mdata: MuData,
        rna_layer: str | None = None,
        junc_ratio_layer: str | None = None,
        atse_counts_layer: str | None = None,
        junc_counts_layer: str | None = None,
        psi_mask_layer: str | None = None,
        batch_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        idx_layer: str | None = None,
        modalities: dict[str, str] | None = None,
        **kwargs,
    ):
        r"""Set up a MuData object for SPLICEVI.

        Parameters
        ----------
        %(param_mdata)s
        rna_layer
            Key in the RNA AnnData for gene expression counts.
            If `None`, the primary data (`.X`) of that AnnData is used.
        junc_ratio_layer
            Key in the splicing AnnData for junction ratio values.
            If `None`, the primary data (`.X`) of that AnnData is used.
        atse_counts_layer
            Key in the splicing AnnData for total event counts.
            If `None`, defaults to `"cell_by_cluster_matrix"`.
        junc_counts_layer
            Key in the splicing AnnData for observed junction counts.
            If `None`, defaults to `"cell_by_junction_matrix"`.
        psi_mask_layer
            Layer with binary mask (1=observed, 0=missing) per junction.
        %(param_batch_key)s
        size_factor_key
            Key in `mdata.obsm` for size factors.
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        %(idx_layer)s
        %(param_modalities)s

        Examples
        --------
        >>> mdata = mu.MuData({
        ...    "rna": ge_anndata.copy(),
        ...    "splicing": atse_anndata.copy()
        ... })
        >>> scvi.model.SPLICEVI.setup_mudata(
        ...     mdata,
        ...     modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"},
        ...     rna_layer="raw_counts",            # gene expression data is in the GE AnnData's "raw_counts" layer
        ...     junc_ratio_layer="junc_ratio",     # splicing data is in the ATSE AnnData's "junc_ratio" layer
        ... )
        >>> model = scvi.model.SPLICEVI(mdata)
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        if modalities is None:
            raise ValueError("Modalities cannot be None.")
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
                psi_mask_layer = "mask"
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
