"""Tests of differential expression and splicing: FDR significance with tied probabilities, the seed argument, and the
errors for unsupported input. Uses the small random MuData of small_data.py, so it runs anywhere (CPU, a few seconds)."""
import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scvi.model.base._de_core import _fdr_de_prediction

from splicevi import SPLICEVI
from splicevi.splicevi import _significant_at_fdr
from small_data import small_mudata


@pytest.fixture(scope="module")
def trained():
    """A model with default settings, trained for two epochs on the small MuData."""
    mdata = small_mudata()
    SPLICEVI.setup_mudata(mdata)
    model = SPLICEVI(mdata)
    model.train(max_epochs=2, accelerator="cpu", batch_size=32)
    return model, mdata


# ----------------------------------------------------------------------------------------------------------------------
# Significance at a target FDR.
# ----------------------------------------------------------------------------------------------------------------------

def test_significance_matches_scvi_without_ties():
    probabilities = pd.Series(np.random.default_rng(1).random(500), index=[f"f{i}" for i in range(500)])
    assert _significant_at_fdr(probabilities, 0.05).equals(_fdr_de_prediction(probabilities, 0.05))


def test_significance_with_ties_keeps_the_count_and_takes_the_first_tied_features():
    # Four features tied at 0.9: at FDR 0.04 the rule keeps 3 features (both 1.0s and one of the 0.9s, since the mean
    # of (1 - probability) is 0.033 for three features and 0.05 for four), and the first 0.9 in the input is the one kept.
    probabilities = pd.Series([0.9, 1.0, 0.9, 0.2, 0.9, 1.0, 0.9], index=list("abcdefg"))
    significant = _significant_at_fdr(probabilities, 0.04)
    assert significant.sum() == _fdr_de_prediction(probabilities, 0.04).sum() == 3
    assert list(significant[significant].index) == ["a", "b", "f"]


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
