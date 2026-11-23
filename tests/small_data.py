"""Small random MuData with the layout SPLICEVI.setup_mudata expects, shared by the tests so they run anywhere
(CPU, no downloads). Import with: from small_data import small_mudata"""
import anndata as ad
import mudata as md
import numpy as np
import scipy.sparse as sp


def small_mudata(n_cells=60, n_genes=50, n_atses=40, junctions_per_atse=3, seed=0):
    """Random MuData with the default layer names of SPLICEVI.setup_mudata.

    Each cell has reads for about 40% of the ATSEs; an ATSE's reads are split over its junctions by a random junction
    usage. obs["group"] is "a" for the first half of the cells and "b" for the second half.
    """
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
