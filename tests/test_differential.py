"""Tests of differential expression and splicing: FDR calls with tied probabilities, the seed argument, and the
errors for unsupported input. Uses a small random MuData built here, so it runs anywhere (CPU, a few seconds)."""
import anndata as ad
import mudata as md
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scvi.model.base._de_core import _fdr_de_prediction

from splicevi import SPLICEVI
from splicevi.splicevi import _fdr_calls


def small_mudata(n_cells=60, n_genes=50, n_atses=40, junctions_per_atse=3, seed=0):
    """Random MuData with the layer names SPLICEVI.setup_mudata expects by default."""
    rng = np.random.default_rng(seed)
    n_junctions = n_atses * junctions_per_atse
    atse_of_junction = np.repeat(np.arange(n_atses), junctions_per_atse)

    # ATSE read totals (60% of cell-ATSE pairs without reads), split over the ATSE's junctions.
    atse_totals = rng.integers(0, 30, size=(n_cells, n_atses)) * (rng.random((n_cells, n_atses)) < 0.4)
    junction_counts = np.zeros((n_cells, n_junctions))
    for cell in range(n_cells):
        for atse in range(n_atses):
            if atse_totals[cell, atse] > 0:
                usage = rng.dirichlet(np.ones(junctions_per_atse))
                junction_counts[cell, atse_of_junction == atse] = rng.multinomial(atse_totals[cell, atse], usage)
    atse_counts = atse_totals[:, atse_of_junction].astype(float)
    psi = np.divide(junction_counts, atse_counts, out=np.zeros_like(junction_counts), where=atse_counts > 0)

    rna = ad.AnnData(X=sp.csr_matrix(rng.poisson(2.0, size=(n_cells, n_genes)).astype(np.float32)))
    rna.layers["length_norm"] = rna.X.copy()

    splicing = ad.AnnData(X=sp.csr_matrix(psi.astype(np.float32)))
    splicing.layers["junc_ratio"] = sp.csr_matrix(psi.astype(np.float32))
    splicing.layers["cell_by_junction_matrix"] = sp.csr_matrix(junction_counts.astype(np.float32))
    splicing.layers["cell_by_cluster_matrix"] = sp.csr_matrix(atse_counts.astype(np.float32))
    splicing.layers["psi_mask"] = sp.csr_matrix((atse_counts > 0).astype(np.float32))
    splicing.var["event_id"] = [f"atse{a}" for a in atse_of_junction]

    rna.obs_names = splicing.obs_names = [f"cell{i}" for i in range(n_cells)]
    mdata = md.MuData({"rna": rna, "splicing": splicing})
    mdata.obsm["X_library_size"] = np.asarray(rna.X.sum(1)).reshape(-1, 1)
    mdata.obs["group"] = np.where(np.arange(n_cells) < n_cells // 2, "a", "b")
    return mdata


@pytest.fixture(scope="module")
def trained():
    """A model with default settings, trained for two epochs on the small MuData."""
    mdata = small_mudata()
    SPLICEVI.setup_mudata(mdata)
    model = SPLICEVI(mdata)
    model.train(max_epochs=2, accelerator="cpu", batch_size=32)
    return model, mdata


# ----------------------------------------------------------------------------------------------------------------------
# FDR calls.
# ----------------------------------------------------------------------------------------------------------------------

def test_fdr_calls_match_scvi_without_ties():
    probabilities = pd.Series(np.random.default_rng(1).random(500), index=[f"f{i}" for i in range(500)])
    assert _fdr_calls(probabilities, 0.05).equals(_fdr_de_prediction(probabilities, 0.05))


def test_fdr_calls_with_ties_keep_the_number_of_calls_and_take_the_first_tied_features():
    # Four features tied at 0.9: at FDR 0.04 the rule allows 3 calls (both 1.0s and one of the 0.9s, since the mean
    # of (1 - probability) is 0.033 for three features and 0.05 for four), and the first 0.9 in the input is called.
    probabilities = pd.Series([0.9, 1.0, 0.9, 0.2, 0.9, 1.0, 0.9], index=list("abcdefg"))
    calls = _fdr_calls(probabilities, 0.04)
    assert calls.sum() == _fdr_de_prediction(probabilities, 0.04).sum() == 3
    assert list(calls[calls].index) == ["a", "b", "f"]


# ----------------------------------------------------------------------------------------------------------------------
# Differential expression and splicing on a trained model.
# ----------------------------------------------------------------------------------------------------------------------

def test_seed_makes_differential_splicing_repeatable(trained):
    model, mdata = trained
    first = model.differential_splicing(mdata, groupby="group", group1="a", group2="b", seed=3, silent=True)
    second = model.differential_splicing(mdata, groupby="group", group1="a", group2="b", seed=3, silent=True)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.index) == list(mdata["splicing"].var_names)


def test_differential_expression_is_sorted_with_ties_in_feature_order(trained):
    model, mdata = trained
    table = model.differential_expression(mdata, groupby="group", group1="a", group2="b", seed=3, silent=True)
    first = model.differential_expression(mdata, groupby="group", group1="a", group2="b", seed=3, silent=True)
    pd.testing.assert_frame_equal(table, first)

    probabilities = table["proba_de"].to_numpy()
    assert np.all(np.diff(probabilities) <= 0)
    gene_position = {name: i for i, name in enumerate(mdata["rna"].var_names)}
    for _, tied in table.groupby("proba_de", sort=False):
        positions = [gene_position[name] for name in tied.index]
        assert positions == sorted(positions)


def test_unknown_splicing_estimate_is_an_error(trained):
    model, mdata = trained
    with pytest.raises(ValueError, match="norm_splicing_function"):
        model.differential_splicing(mdata, groupby="group", group1="a", group2="b", norm_splicing_function="mean")


def test_setup_anndata_points_to_setup_mudata():
    with pytest.raises(NotImplementedError, match="setup_mudata"):
        SPLICEVI.setup_anndata(ad.AnnData(np.zeros((3, 3))))
