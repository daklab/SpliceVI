"""
MeanBayes: Bayesian PSI imputation baseline using Beta conjugate priors.

This module provides a simple, model-free baseline for imputing missing PSI
(percent-spliced-in) values in single-cell splicing AnnData objects.  It uses
method-of-moments to fit a Beta prior from the observed PSI distribution and
then computes the conjugate posterior mean for every masked position.

An optional grouping field (e.g. cell type) allows per-group priors so that
the imputation is conditioned on cell-type-specific PSI landscapes rather than
the global one.

Typical usage
-------------
>>> mb = MeanBayes(adata, group_by="broad_cell_type")
>>> imputed_df = mb.get_imputed_splicing()      # returns full cells × junctions DataFrame
>>> imputed_arr = mb.get_imputed_splicing(return_numpy=True)
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────


def _to_dense(mat) -> np.ndarray:
    """Convert a sparse or dense matrix to a 2-D float64 ndarray."""
    if sp.issparse(mat):
        return np.asarray(mat.todense(), dtype=np.float64)
    return np.asarray(mat, dtype=np.float64)


def _fit_beta_prior(
    psi: np.ndarray,
    observed_mask: np.ndarray,
) -> tuple[float, float]:
    """Fit a Beta(a, b) prior from observed PSI values using method of moments.

    The prior is derived from the *junction-level* distribution of mean PSI
    values: for each junction we compute the mean over observed cells, then fit
    Beta parameters to those means.

    Parameters
    ----------
    psi:
        Dense array of shape ``(n_cells, n_junctions)`` with PSI values.
        Unobserved positions can hold any value — they are ignored via
        ``observed_mask``.
    observed_mask:
        Boolean array of shape ``(n_cells, n_junctions)``.  ``True`` means the
        PSI value for that (cell, junction) pair was actually measured.

    Returns
    -------
    a_prior, b_prior:
        Concentration parameters of the fitted Beta prior.  Falls back to
        ``Beta(1, 1)`` (uniform) when there is insufficient data.
    """
    n_cells, n_junctions = psi.shape

    junction_means = np.empty(n_junctions)
    for j in range(n_junctions):
        observed_vals = psi[:, j][observed_mask[:, j]]
        junction_means[j] = observed_vals.mean() if len(observed_vals) > 0 else 0.5

    # Global mean and variance of the per-junction means
    global_mean = junction_means.mean()
    global_var = junction_means.var()

    # Degenerate case: all junction means identical → no information → uniform prior
    max_var = global_mean * (1.0 - global_mean)
    if max_var <= 1e-8 or global_var <= 1e-8:
        logger.debug("Degenerate prior variance — falling back to Beta(1, 1)")
        return 1.0, 1.0

    # Clip variance so the method-of-moments formula stays well-defined
    global_var = np.clip(global_var, 1e-8, max_var - 1e-8)

    # Beta method-of-moments: common = mean*(1-mean)/var - 1
    common = (global_mean * (1.0 - global_mean) / global_var) - 1.0
    a_prior = global_mean * common
    b_prior = (1.0 - global_mean) * common

    # Guard against numerically negative concentrations
    a_prior = max(a_prior, 1e-6)
    b_prior = max(b_prior, 1e-6)

    logger.debug("Fitted Beta prior: a=%.4f  b=%.4f", a_prior, b_prior)
    return float(a_prior), float(b_prior)


def _impute_with_prior(
    psi: np.ndarray,
    observed_mask: np.ndarray,
    a_prior: float,
    b_prior: float,
) -> np.ndarray:
    """Compute the Beta-conjugate posterior mean for every masked position.

    Observed PSI values are treated as a sufficient statistic of a Beta-Binomial
    likelihood.  With observed sum ``S`` and count ``N`` for a given junction,
    the posterior mean under a ``Beta(a, b)`` prior is:

    .. math::
        \\hat{\\psi} = \\frac{a + S}{a + b + N}

    Parameters
    ----------
    psi:
        Dense ``(n_cells, n_junctions)`` PSI matrix.
    observed_mask:
        Boolean ``(n_cells, n_junctions)`` mask; ``True`` = observed.
    a_prior, b_prior:
        Beta prior concentrations (fitted by :func:`_fit_beta_prior`).

    Returns
    -------
    imputed:
        Dense ``(n_cells, n_junctions)`` array.  Observed entries are copied
        unchanged; masked entries are replaced by the posterior mean.
    """
    # Per-junction sufficient statistics from observed cells only
    # psi_masked_zero: zero-out unobserved so sum/count are over observed only
    psi_obs_only = psi * observed_mask  # broadcast: unobserved → 0

    sum_psi = psi_obs_only.sum(axis=0)   # (n_junctions,)
    n_obs = observed_mask.sum(axis=0)    # (n_junctions,) int counts

    # Posterior mean per junction: shape (n_junctions,)
    post_mean_per_junction = (a_prior + sum_psi) / (a_prior + b_prior + n_obs)

    # Observed → keep psi value, masked → posterior mean (broadcasts over all cells)
    imputed = np.where(observed_mask, psi, post_mean_per_junction[np.newaxis, :])

    return imputed


# ──────────────────────────────────────────────────────────────────────────────
# Public class
# ──────────────────────────────────────────────────────────────────────────────


class MeanBayes:
    """Bayesian PSI imputation using a Beta conjugate prior.

    This class provides a non-parametric, model-free baseline for imputing
    missing PSI values in a splicing-only AnnData object.  For each junction,
    a Beta prior is fitted from the observed PSI distribution via method of
    moments.  Missing entries are then replaced by the posterior mean under
    that prior.

    When ``group_by`` is provided, priors are estimated *per group* (e.g. per
    cell type), so the imputed value for a masked (cell, junction) pair reflects
    the PSI landscape of that cell's group rather than the global average.

    Parameters
    ----------
    adata:
        AnnData object whose ``X`` or ``layers[junc_ratio_layer]`` contains PSI
        values and whose ``layers[psi_mask_layer]`` contains a binary mask
        (``1`` = observed, ``0`` = masked / missing).
    junc_ratio_layer:
        Name of the AnnData layer holding PSI values.  Defaults to
        ``"junc_ratio"`` to match the SpliceVI / EDDISPLICE convention.
    psi_mask_layer:
        Name of the AnnData layer holding the binary observation mask.
        Defaults to ``"psi_mask"``.
    group_by:
        Optional column name in ``adata.obs``.  When supplied, Beta priors are
        fitted independently for each unique value (e.g. each cell type).
        Cells whose group has fewer than ``min_group_cells`` observed cells for
        a given junction fall back to the global prior for that junction.
    min_group_cells:
        Minimum number of observed cells required in a group before using a
        group-specific prior.  Groups with fewer cells fall back to the global
        prior.  Default is ``2``.

    Examples
    --------
    >>> mb = MeanBayes(adata)
    >>> imputed_df = mb.get_imputed_splicing()

    >>> mb_ct = MeanBayes(adata, group_by="cell_type")
    >>> imputed_df = mb_ct.get_imputed_splicing(indices=[0, 1, 2])
    """

    def __init__(
        self,
        adata: AnnData,
        junc_ratio_layer: str = "junc_ratio",
        psi_mask_layer: str = "psi_mask",
        group_by: str | None = None,
        min_group_cells: int = 2,
    ) -> None:
        self._validate_inputs(adata, junc_ratio_layer, psi_mask_layer, group_by)

        self.adata = adata
        self.junc_ratio_layer = junc_ratio_layer
        self.psi_mask_layer = psi_mask_layer
        self.group_by = group_by
        self.min_group_cells = min_group_cells

        # Eagerly extract dense arrays so repeated calls to get_imputed_splicing
        # do not re-extract from the AnnData every time.
        self._psi: np.ndarray = _to_dense(adata.layers[junc_ratio_layer])
        self._mask: np.ndarray = _to_dense(adata.layers[psi_mask_layer]).astype(bool)

        logger.info(
            "MeanBayes initialised: %d cells × %d junctions  |  "
            "group_by=%s  |  %.1f%% entries observed",
            adata.n_obs,
            adata.n_vars,
            group_by,
            100.0 * self._mask.mean(),
        )

    # ── Validation ────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_inputs(
        adata: AnnData,
        junc_ratio_layer: str,
        psi_mask_layer: str,
        group_by: str | None,
    ) -> None:
        """Raise informative errors for common mis-configurations."""
        if not isinstance(adata, AnnData):
            raise TypeError(f"adata must be an AnnData object, got {type(adata)}")
        if junc_ratio_layer not in adata.layers:
            raise KeyError(
                f"Layer '{junc_ratio_layer}' not found in adata.layers.  "
                f"Available layers: {list(adata.layers.keys())}"
            )
        if psi_mask_layer not in adata.layers:
            raise KeyError(
                f"Layer '{psi_mask_layer}' not found in adata.layers.  "
                f"Available layers: {list(adata.layers.keys())}"
            )
        if group_by is not None and group_by not in adata.obs.columns:
            raise KeyError(
                f"group_by column '{group_by}' not found in adata.obs.  "
                f"Available columns: {list(adata.obs.columns)}"
            )

    # ── Core computation ──────────────────────────────────────────────────────

    def _compute_imputed_global(
        self,
        psi: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Impute using a single Beta prior fitted from all cells.

        Parameters
        ----------
        psi:
            Dense PSI matrix, shape ``(n_cells, n_junctions)``.
        mask:
            Boolean observed mask, same shape as ``psi``.

        Returns
        -------
        np.ndarray of shape ``(n_cells, n_junctions)``.
        """
        a, b = _fit_beta_prior(psi, mask)
        return _impute_with_prior(psi, mask, a, b)

    def _compute_imputed_by_group(
        self,
        psi: np.ndarray,
        mask: np.ndarray,
        groups: np.ndarray,
    ) -> np.ndarray:
        """Impute using per-group Beta priors.

        For each group (e.g. cell type), a separate Beta prior is fitted from
        the cells belonging to that group.  If a group contains fewer than
        ``self.min_group_cells`` observed values for a junction, the global
        prior is used for that junction instead.

        Parameters
        ----------
        psi:
            Dense PSI matrix, shape ``(n_cells, n_junctions)``.
        mask:
            Boolean observed mask, same shape as ``psi``.
        groups:
            1-D array of group labels, length ``n_cells``.

        Returns
        -------
        np.ndarray of shape ``(n_cells, n_junctions)``.
        """
        n_cells, n_junctions = psi.shape
        imputed = np.empty_like(psi)

        # Global fallback prior — used for junctions with sparse group coverage
        a_global, b_global = _fit_beta_prior(psi, mask)

        # Per-junction global sufficient statistics (for fallback)
        sum_psi_global = (psi * mask).sum(axis=0)   # (n_junctions,)
        n_obs_global = mask.sum(axis=0)              # (n_junctions,)

        unique_groups = np.unique(groups)
        logger.info("Computing per-group priors for %d groups", len(unique_groups))

        for grp in unique_groups:
            cell_idx = np.where(groups == grp)[0]
            psi_grp = psi[cell_idx]    # (n_grp_cells, n_junctions)
            mask_grp = mask[cell_idx]  # (n_grp_cells, n_junctions)

            # Fit group-level Beta prior from cells in this group
            a_grp, b_grp = _fit_beta_prior(psi_grp, mask_grp)

            # Per-junction sufficient statistics within this group
            sum_psi_grp = (psi_grp * mask_grp).sum(axis=0)  # (n_junctions,)
            n_obs_grp = mask_grp.sum(axis=0)                 # (n_junctions,)

            # Decide per junction: use group prior if enough observations,
            # otherwise fall back to the global prior
            use_group = n_obs_grp >= self.min_group_cells  # (n_junctions,) bool

            # Posterior mean per junction under group prior
            post_mean_grp = (a_grp + sum_psi_grp) / (a_grp + b_grp + n_obs_grp)

            # Posterior mean per junction under global prior
            post_mean_glob = (a_global + sum_psi_global) / (
                a_global + b_global + n_obs_global
            )

            # Select which posterior mean to use per junction
            post_mean = np.where(use_group, post_mean_grp, post_mean_glob)
            # post_mean shape: (n_junctions,)

            # Fill imputed values for cells in this group:
            #   observed → keep original PSI,  masked → posterior mean
            grp_imputed = np.where(mask_grp, psi_grp, post_mean[np.newaxis, :])
            imputed[cell_idx] = grp_imputed

            logger.debug(
                "Group '%s': %d cells, a=%.4f, b=%.4f, "
                "%d/%d junctions used group prior",
                grp,
                len(cell_idx),
                a_grp,
                b_grp,
                int(use_group.sum()),
                n_junctions,
            )

        return imputed

    # ── Public API ────────────────────────────────────────────────────────────

    def get_imputed_splicing(
        self,
        indices: Sequence[int] | None = None,
        junction_list: Sequence[str] | None = None,
        return_numpy: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        """Return the imputed PSI matrix.

        Observed PSI values are returned unchanged.  Masked positions (where
        ``psi_mask == 0``) are replaced by the Beta posterior mean computed from
        the observed data.

        When the model was initialised with ``group_by``, the prior is estimated
        separately for each group so that imputed values reflect the typical PSI
        level within that cell's group (e.g. its cell type) rather than the
        global mean.

        Parameters
        ----------
        indices:
            Integer cell indices to include in the output.  If ``None``, all
            cells are returned.  Posterior computation always uses *all* cells
            to ensure stable prior estimation; only the returned rows are
            filtered.
        junction_list:
            Subset of junction names (matching ``adata.var_names``) to include
            in the output.  If ``None``, all junctions are returned.
        return_numpy:
            If ``True``, returns a raw ``np.ndarray`` instead of a
            ``pd.DataFrame``.

        Returns
        -------
        np.ndarray or pd.DataFrame of shape ``(n_selected_cells, n_selected_junctions)``.
        The imputed matrix has the same scale as the input PSI values (i.e.
        values in ``[0, 1]``).

        Notes
        -----
        The posterior mean for junction *j* given a ``Beta(a, b)`` prior and
        ``N`` observed cells with sum ``S`` is:

        .. math::
            \\hat{\\psi}_j = \\frac{a + S}{a + b + N}

        This shrinks the empirical mean towards the prior mean ``a / (a + b)``
        in proportion to how many cells were observed.
        """
        psi = self._psi   # (n_cells, n_junctions)
        mask = self._mask

        # ── Compute imputed matrix (always over all cells for stable prior) ──
        if self.group_by is not None:
            groups = self.adata.obs[self.group_by].values
            imputed = self._compute_imputed_by_group(psi, mask, groups)
        else:
            imputed = self._compute_imputed_global(psi, mask)

        # ── Filter to requested cells ──────────────────────────────────────
        if indices is None:
            indices = np.arange(self.adata.n_obs)
        else:
            indices = np.asarray(indices)
        imputed = imputed[indices]

        # ── Filter to requested junctions ─────────────────────────────────
        var_names = self.adata.var_names
        if junction_list is not None:
            junc_set = set(junction_list)
            col_mask = np.array([v in junc_set for v in var_names])
            imputed = imputed[:, col_mask]
            col_names = var_names[col_mask]
        else:
            col_names = var_names

        if return_numpy:
            return imputed

        return pd.DataFrame(
            imputed,
            index=self.adata.obs_names[indices],
            columns=col_names,
        )
