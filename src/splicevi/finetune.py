"""Fine-tune a trained SpliceVI model's decoders on new data, keeping its encoders (and so its latent space) fixed.

The finetuning protocol of the paper's EasySci model: a model trained on a deep reference atlas (Allen Brain, Smart-seq) is
adapted to a sparse query atlas (EasySci, single-nucleus 3') by training only its two decoders plus one zero-initialized
decoder covariate input (technology: reference 0, query 1), on the query cells together with a replay set of reference
cells so the decoders keep fitting the reference.

    model = SPLICEVI.load(reference_model_dir, adata=query_mdata)
    history = model.finetune_decoders(query_mdata, reference_mdata)
    model.save(out_dir)          # reloads with SPLICEVI.load(out_dir, adata=...), decoding as query data

What is trained and what is kept from the reference model:
- trained: both decoders in full (expression: hidden layers, scale head incl. per-gene offsets, zero-inflation head;
  splicing: hidden layers and output layer) and the decoder covariate weights;
- kept: the expression and splicing encoders, junction embeddings and latent mixing (so query cells are placed in the
  reference latent space), the population PSI and pool divisor of the deviation input, the per-ATSE concentrations and
  the gene dispersions. Re-learning the concentrations and dispersions on the query data was tested and
    predicted held-out query reads worse, so it is not offered here.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import torch
from scvi import REGISTRY_KEYS

TRAINABLE_PREFIXES = ("z_decoder_expression.", "z_decoder_splicing.", "decoder_covariate_weight_")


def log(message):
    print(f"[FINETUNE_DECODERS] {message}", flush=True)


def _register(model, mdata):
    """Register mdata with the model the way SPLICEVI.load does: setup_mudata with the model's own setup arguments
    (source_registry = the model's registry) if it was never set up, then transfer the model's fields to it.
    Views are copied first (setup writes obs["_indices"])."""
    from scvi.data._constants import _SETUP_ARGS_KEY
    if mdata.is_view:
        mdata = mdata.copy()
    if model.get_anndata_manager(mdata) is None:
        registry = model.registry_
        type(model).setup_mudata(mdata, source_registry=registry, **registry[_SETUP_ARGS_KEY])
    return model._validate_anndata(mdata)


def finetune_decoders(
    model,
    query_mdata,
    reference_mdata,
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
    """See :meth:`splicevi.SPLICEVI.finetune_decoders` (the documented entry point)."""
    module = model.module
    device = module.device

    # check everything before the model is changed
    if module.n_decoder_covariates not in (0, 1):
        raise ValueError(f"finetune_decoders uses one decoder covariate; this model has {module.n_decoder_covariates}")
    module.decoder_first_layers()   # raises NotImplementedError for linear decoders
    n_query, n_reference = query_mdata.n_obs, reference_mdata.n_obs
    n_validation = int(round(validation_fraction * n_query))
    if n_validation < 1 or n_validation >= n_query:
        raise ValueError(f"validation_fraction = {validation_fraction} of {n_query} query cells gives {n_validation} validation "
                         "cells; need at least 1 validation and 1 training cell")
    if n_reference < n_reference_per_step:
        raise ValueError(f"reference_mdata has {n_reference} cells, fewer than n_reference_per_step = {n_reference_per_step}")

    query_mdata = _register(model, query_mdata)
    reference_mdata = _register(model, reference_mdata)
    if set_torch_seed:
        torch.manual_seed(seed)   # global torch RNG: latent sampling and decoder dropout during fine-tuning
    rng = np.random.default_rng(seed)   # validation split, minibatch order, reference cells per step

    # model state for the query data (persistent buffers, saved with the model)
    if module.n_decoder_covariates == 0:
        model.add_decoder_covariates(1)
    if fix_depth_inputs and getattr(module, "decoder_depth_covariates", False):
        with torch.no_grad():   # standardized depth inputs = 0, i.e. the reference training mean, for every cell
            module.depth_cov_std.fill_(float("inf"))
    if hasattr(module, "set_cross_gate"):
        module.set_cross_gate(1.0)   # cross-modality gate open, as at the end of training

    perm = rng.permutation(n_query)
    validation_idx, train_idx = np.sort(perm[:n_validation]), perm[n_validation:]

    def minibatch(mdata, idx):
        dl = model._make_data_loader(adata=mdata, indices=idx, batch_size=len(idx), shuffle=False)
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in next(iter(dl)).items()}

    def batch_loss(tensors, covariate_value):
        n = tensors[REGISTRY_KEYS.X_KEY].shape[0]
        module.decoder_covariate_batch = torch.full((n, 1), float(covariate_value), device=device)
        try:
            inference_outputs = module.inference(**module._get_inference_input(tensors))
            generative_outputs = module.generative(**module._get_generative_input(tensors, inference_outputs))
            return module.loss(tensors, inference_outputs, generative_outputs).loss
        finally:
            module.decoder_covariate_batch = None

    def set_mode(train):
        module.eval()   # encoders stay in eval mode (frozen)
        if train:
            module.z_decoder_expression.train()
            module.z_decoder_splicing.train()

    def train_epoch(optimizer):
        set_mode(True)
        total = 0.0
        order = rng.permutation(train_idx)
        for start in range(0, len(order), n_query_per_step):
            query_idx = np.sort(order[start:start + n_query_per_step])
            reference_idx = np.sort(rng.choice(n_reference, n_reference_per_step, replace=False))
            loss = (batch_loss(minibatch(query_mdata, query_idx), query_covariate_value)
                    + batch_loss(minibatch(reference_mdata, reference_idx), reference_covariate_value))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss)
        return total

    def validation_loss():
        set_mode(False)
        total, n = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(validation_idx), n_query_per_step):
                idx = validation_idx[start:start + n_query_per_step]
                total += float(batch_loss(minibatch(query_mdata, idx), query_covariate_value)) * len(idx)
                n += len(idx)
        return total / n

    def snapshot():
        return {k: v.detach().clone() for k, v in module.state_dict().items()}

    # only the decoders and the covariate weights train; flags restored whatever happens
    requires_grad_before = {name: p.requires_grad for name, p in module.named_parameters()}
    rows, t0 = [], time.time()
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        trainable = [p for name, p in module.named_parameters() if name.startswith(TRAINABLE_PREFIXES)]
        for p in trainable:
            p.requires_grad_(True)
        optimizer = torch.optim.Adam(trainable, lr=lr)
        log(f"query {n_query:,} cells ({len(train_idx):,} training, {n_validation:,} validation); reference replay "
            f"{n_reference:,} cells; {n_query_per_step} query + {n_reference_per_step} reference cells per step; trainable "
            f"{sum(p.numel() for p in trainable):,} parameters (decoders + decoder covariate weights); seed {seed}")

        start_validation = validation_loss()
        rows.append(dict(epoch=-1, train_loss=np.nan, validation_loss=start_validation))
        log(f"epoch -1: validation {start_validation:.3f} (decoder covariate weights zero = the reference model)")
        best, best_epoch, wait, best_state = np.inf, -1, 0, snapshot()
        stop_reason = f"max_epochs ({max_epochs})"
        for epoch in range(max_epochs):
            train_loss, val_loss = train_epoch(optimizer), validation_loss()
            rows.append(dict(epoch=epoch, train_loss=train_loss, validation_loss=val_loss))
            if val_loss < best:
                best, best_epoch, wait, best_state = val_loss, epoch, 0, snapshot()
                if checkpoint_dir:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(best_state, os.path.join(checkpoint_dir, "module_state_best.pt"))
            else:
                wait += 1
            if epoch % 10 == 0 or wait == 0:
                log(f"epoch {epoch}: train {train_loss:.1f}, validation {val_loss:.3f} (best {best:.3f} at epoch {best_epoch})")
            if wait >= early_stopping_patience:
                stop_reason = f"early stopping at epoch {epoch} (patience {early_stopping_patience})"
                break
            if max_minutes is not None and (time.time() - t0) / 60 > max_minutes:
                stop_reason = f"max_minutes ({max_minutes}) reached at epoch {epoch}"
                break
        module.load_state_dict(best_state)   # best state, then query mode
        with torch.no_grad():
            module.decoder_covariate_value.fill_(float(query_covariate_value))
    finally:
        module.decoder_covariate_batch = None
        for name, p in module.named_parameters():
            p.requires_grad_(requires_grad_before.get(name, True))
        module.eval()

    history = pd.DataFrame(rows)
    model.finetune_history_ = history
    model.finetune_params_ = dict(
        n_query_per_step=n_query_per_step, n_reference_per_step=n_reference_per_step, lr=lr, max_epochs=max_epochs,
        early_stopping_patience=early_stopping_patience, validation_fraction=validation_fraction, seed=seed,
        set_torch_seed=set_torch_seed, max_minutes=max_minutes, fix_depth_inputs=fix_depth_inputs,
        query_covariate_value=query_covariate_value, reference_covariate_value=reference_covariate_value,
        n_query=n_query, n_reference=n_reference, best_epoch=best_epoch, best_validation_loss=float(best),
        start_validation_loss=float(start_validation), stop_reason=stop_reason,
        minutes=round((time.time() - t0) / 60, 1))
    log(f"{stop_reason}; best validation {best:.3f} at epoch {best_epoch} (start {start_validation:.3f}); "
        f"{model.finetune_params_['minutes']} min; model now decodes as query data (decoder covariate = {query_covariate_value})")
    return history
