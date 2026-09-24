"""Gradient budget diagnostic (STAGE5 section 63). On our branch only; not in Smriti's files.

Per checkpoint, no optimizer step: K fixed minibatches (seeded cell indices, the same for every
model), train mode with dropout OFF, cross gate open (mixing live). For each of the three terms as
they enter the training objective -- L_GE = mean over cells of the masked ZINB reconstruction,
L_SP = mean of the DM reconstruction x splicing_loss_weight, KL = mean of kl_weight*KL(z) + the
paired penalty; loss() = mean(sum of these per cell) + phi prior -- the gradient is taken with
torch.autograd.grad on (a) the splicing encoder's parameters and (b) the expression encoder's.
Reported per encoder, per batch: the L2 norm of each term's gradient, R_s = ||dL_GE/dth_s|| /
||dL_SP/dth_s||, R_e = ||dL_SP/dth_e|| / ||dL_GE/dth_e||, and the cosine between dL_GE and dL_SP;
plus the norms on the joint latent mean for continuity with the 2026-09 measurement (STAGE4 4.4).
"""
import os
import numpy as np
import pandas as pd
import torch

ENCODERS = {"splicing": "z_encoder_splicing", "expression": "z_encoder_expression"}
TERMS = ("L_GE", "L_SP", "KL")


def fixed_batches(n_obs: int, K: int = 20, batch: int = 512, seed: int = 20260924) -> np.ndarray:
    """K x batch cell indices, one seeded permutation; identical for every model of the same n_obs."""
    rng = np.random.default_rng(seed)
    return rng.permutation(n_obs)[: K * batch].reshape(K, batch)


def _set_dropout(module, enabled: bool):
    for m in module.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train(enabled)


def _params(module, prefix):
    return [p for n, p in module.named_parameters() if n.startswith(prefix) and p.requires_grad]


def _flat_grad(term, params):
    gs = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(gs, params)])


def objective_terms(module, tensors, kl_weight: float = 1.0):
    """The three terms exactly as loss() combines them (taken from loss()'s own outputs)."""
    inf = module.inference(**module._get_inference_input(tensors))
    gen = module.generative(**module._get_generative_input(tensors, inf))
    out = module.loss(tensors, inf, gen, kl_weight=kl_weight)
    rl, kl = out.reconstruction_loss, out.kl_local
    terms = {"L_GE": rl["reconstruction_loss_expression"].mean(),
             "L_SP": rl["reconstruction_loss_splicing"].mean(),
             "KL": (kl_weight * kl["kl_divergence_z"] + kl["kl_divergence_paired"]).mean()}
    return terms, inf


def budget_on_batch(module, tensors, kl_weight: float = 1.0) -> dict:
    terms, inf = objective_terms(module, tensors, kl_weight)
    rec = {f"loss_{k}": float(v) for k, v in terms.items()}
    for enc, prefix in ENCODERS.items():
        params = _params(module, prefix)
        g = {k: _flat_grad(v, params) for k, v in terms.items()}
        for k in TERMS:
            rec[f"{enc}_norm_{k}"] = float(g[k].norm())
        rec[f"{enc}_cos_GE_SP"] = float(torch.dot(g["L_GE"], g["L_SP"]) / (g["L_GE"].norm() * g["L_SP"].norm() + 1e-12))
    rec["R_s"] = rec["splicing_norm_L_GE"] / (rec["splicing_norm_L_SP"] + 1e-12)
    rec["R_e"] = rec["expression_norm_L_SP"] / (rec["expression_norm_L_GE"] + 1e-12)
    qz = inf["qz_m"]
    for k, v in terms.items():
        gz = torch.autograd.grad(v, qz, retain_graph=True, allow_unused=True)[0]
        rec[f"z_norm_{k}"] = float(gz.norm()) if gz is not None else 0.0
    return rec


def gradient_budget(model, mdata, indices: np.ndarray, batch: int = 512, kl_weight: float = 1.0,
                    dropout: bool = False, cross_gate: float = 1.0) -> list[dict]:
    """Run budget_on_batch over the fixed batches; restores the module's mode and gate afterwards."""
    module = model.module
    was_training = module.training
    gate0 = float(module.cross_gate.item()) if hasattr(module, "cross_gate") else None
    module.train(); _set_dropout(module, dropout)
    if hasattr(module, "set_cross_gate"):
        module.set_cross_gate(cross_gate)
    recs = []
    for b in range(indices.shape[0]):
        dl = model._make_data_loader(adata=mdata, indices=indices[b], batch_size=batch, shuffle=False)
        tensors = next(iter(dl))
        tensors = {k: (v.to(module.device) if torch.is_tensor(v) else v) for k, v in tensors.items()}
        with torch.enable_grad():
            recs.append(budget_on_batch(module, tensors, kl_weight))
    module.zero_grad(set_to_none=True)
    if gate0 is not None:
        module.set_cross_gate(gate0)
    module.train(was_training)
    return recs


def summarise(recs: list[dict]) -> dict:
    df = pd.DataFrame(recs); mean, sd = df.mean(), df.std(ddof=1)
    out = {}
    for k in df.columns:
        out[f"{k}_mean"] = float(mean[k]); out[f"{k}_sd"] = float(sd[k])
    out["n_batches"] = len(df)
    return out


try:
    from lightning.pytorch.callbacks import Callback
except Exception:  # pragma: no cover
    from pytorch_lightning.callbacks import Callback


class GradientBudgetCallback(Callback):
    """Log the budget every `every` epochs on the same fixed batches (flag-gated in train_splicevi.py)."""

    def __init__(self, model, mdata, out_path: str, every: int = 50, K: int = 20, batch: int = 512, seed: int = 20260924):
        self.model, self.mdata, self.out, self.every, self.batch = model, mdata, out_path, every, batch
        self.indices = fixed_batches(mdata.n_obs, K, batch, seed)

    def on_train_epoch_end(self, trainer, pl_module):
        ep = trainer.current_epoch + 1
        if ep % self.every:
            return
        s = summarise(gradient_budget(self.model, self.mdata, self.indices, self.batch))
        s.update(epoch=ep)
        pd.DataFrame([s]).to_csv(self.out, mode="a", header=not os.path.exists(self.out), index=False, sep="\t")
        print(f"[GRADIENT_BUDGET] epoch {ep}: R_s={s['R_s_mean']:.3f} cos_s={s['splicing_cos_GE_SP_mean']:.3f} "
              f"R_e={s['R_e_mean']:.3f} cos_e={s['expression_cos_GE_SP_mean']:.3f}", flush=True)
