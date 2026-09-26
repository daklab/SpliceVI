"""STAGE5 §85 / §86.3: "one vote per event" splicing encoder, votes grouped by gene.

Each observed alternative-splicing event (ATSE) contributes ONE token: the embedding of its winning junction (the junction with the most reads;
with about one molecule per cell-event, STAGE5 §77, that is the observation). Tokens of events from the same gene are averaged into one gene vote
(same-gene events co-vary, §77c z 15, so they are one vote, not several). The cell representation is the mean over genes of MLP(gene vote), mapped
to (mu, raw_logvar) by the same cell-level MLP design as PartialEncoderEDDIFaster, so the SPLICEVAE partial-encoder code path is reused unchanged.

Inputs match PartialEncoderEDDIFaster.forward(x, mask, *cat_list, cont=None, weights=None):
  x    (B, J) junction usage ratios (k / n); the winner of an event is argmax_j of x over its observed junctions (tie -> lowest junction index);
  mask (B, J) 1 = observed junction.
Buffers junc_event (J,) and junc_gene (J,) (long) are filled from the training AnnData by SPLICEVI.init_token_index_from_adata(); they are saved in
the state dict. Nothing about the fraction beyond which junction won reaches the network.
"""
from __future__ import annotations
from collections.abc import Iterable
import torch
import torch.nn as nn
import torch.nn.functional as F
from scvi.nn import FCLayers


class TokenVoteEncoder(nn.Module):
    def __init__(self, input_dim: int, code_dim: int, h_hidden_dim: int, encoder_hidden_dim: int, latent_dim: int, dropout_rate: float = 0.0,
                 n_cat_list: Iterable[int] | None = None, n_cont: int = 0, inject_covariates: bool = True, encoder_n_layers: int = 2,
                 embedding_init_sd: float = 0.1, **_ignored):
        super().__init__()
        self.code_dim = code_dim
        self.register_buffer("junc_event", torch.zeros(input_dim, dtype=torch.long))
        self.register_buffer("junc_gene", torch.zeros(input_dim, dtype=torch.long))
        self.register_buffer("index_set", torch.zeros((), dtype=torch.bool))
        self.feature_embedding = nn.Parameter(torch.randn(input_dim, code_dim) * embedding_init_sd)   # random init (68.12: the SVD table is not an asset)
        self.gene_layer = nn.Sequential(nn.Linear(code_dim, h_hidden_dim), nn.LayerNorm(h_hidden_dim), nn.ReLU(), nn.Dropout(dropout_rate),
                                        nn.Linear(h_hidden_dim, code_dim), nn.LayerNorm(code_dim), nn.ReLU())
        self.encoder_mlp = FCLayers(n_in=code_dim, n_out=2 * latent_dim, n_cat_list=n_cat_list or [], n_cont=n_cont, n_layers=encoder_n_layers,
                                    n_hidden=encoder_hidden_dim, dropout_rate=dropout_rate, use_batch_norm=False, use_layer_norm=True,
                                    inject_covariates=inject_covariates)
        self.z_transformation = lambda v: v

    @torch.no_grad()
    def set_index(self, junc_event: torch.Tensor, junc_gene: torch.Tensor) -> None:
        self.junc_event.copy_(junc_event.to(self.junc_event)); self.junc_gene.copy_(junc_gene.to(self.junc_gene)); self.index_set.fill_(True)

    @torch.no_grad()
    def winners(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(b_idx, j_idx) of the winning junction of every observed (cell, event)."""
        if not bool(self.index_set):
            raise RuntimeError("TokenVoteEncoder: junc_event / junc_gene not set (call SPLICEVI.init_token_index_from_adata)")
        B, J = x.shape; n_ev = int(self.junc_event.max()) + 1
        obs = mask > 0
        ev = self.junc_event.expand(B, J)
        score = torch.where(obs, x.float(), torch.full_like(x, float("-inf"), dtype=torch.float32))          # unobserved -> -inf
        ev_max = torch.full((B, n_ev), float("-inf"), device=x.device).scatter_reduce(1, ev, score, reduce="amax", include_self=True)
        win = obs & (score == ev_max.gather(1, ev))
        # ties: keep the lowest junction index among the maxima of each (cell, event)
        jr = torch.arange(J, device=x.device).expand(B, J)
        cand = torch.where(win, jr, torch.full_like(jr, J))
        first = torch.full((B, n_ev), J, device=x.device, dtype=jr.dtype).scatter_reduce(1, ev, cand, reduce="amin", include_self=True)
        win = win & (jr == first.gather(1, ev))
        return win.nonzero(as_tuple=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, *cat_list: torch.Tensor, cont: torch.Tensor | None = None, weights: torch.Tensor | None = None):
        B = x.shape[0]; D = self.code_dim; dev = x.device
        b_idx, j_idx = self.winners(x, mask)
        if b_idx.numel() == 0:
            mu_logvar = self.encoder_mlp(torch.zeros(B, D, device=dev), *cat_list, cont=cont); return mu_logvar.chunk(2, dim=-1)
        tok = F.normalize(self.feature_embedding, p=2, dim=1, eps=1e-8).index_select(0, j_idx)            # (T, D) one token per observed event
        n_gene = int(self.junc_gene.max()) + 1
        key = b_idx * n_gene + self.junc_gene[j_idx]                                                        # (cell, gene)
        uk, inv = torch.unique(key, return_inverse=True)
        gsum = torch.zeros(uk.numel(), D, device=dev, dtype=tok.dtype).index_add_(0, inv, tok)
        gcnt = torch.zeros(uk.numel(), 1, device=dev, dtype=tok.dtype).index_add_(0, inv, torch.ones_like(tok[:, :1]))
        gvote = gsum / gcnt                                                                                 # one vote per (cell, gene)
        h = self.gene_layer(gvote)
        gb = torch.div(uk, n_gene, rounding_mode="floor")
        pooled = torch.zeros(B, D, device=dev, dtype=h.dtype).index_add_(0, gb, h)
        pooled = pooled / torch.bincount(gb, minlength=B).to(h.dtype).view(B, 1).clamp_min(1)
        mu_logvar = self.encoder_mlp(pooled, *cat_list, cont=cont)
        mu, logvar = mu_logvar.chunk(2, dim=-1)
        return mu, logvar
