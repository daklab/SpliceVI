# SpliceVI MuData Object

SpliceVI takes a [MuData](https://mudata.readthedocs.io/) object as input, with two modalities keyed `"rna"` and `"splicing"`. This page describes the required structure of each modality and how the MuData is assembled from raw inputs.

---

## Overview

```
MuData
├── mdata["rna"]       AnnData   (cells × genes)   — gene expression
└── mdata["splicing"]  AnnData   (cells × junctions) — splicing
```

Both modalities share the same `obs` (cell barcodes / metadata) index. All cells must be present in both modalities.

---

## RNA modality — `mdata["rna"]`

| Slot | Contents |
|------|----------|
| `.X` | Raw gene expression counts `(cells × genes)`, integer dtype |
| `.layers["length_norm"]` | Length-normalized expression: raw counts divided by transcript length and rescaled to the median transcript length across genes. Used as the model's expression input. |
| `.var["gene_id"]` | Ensembl gene ID |
| `.var["modality"]` | `"Gene_Expression"` |
| `.obsm["X_library_size"]` | Per-cell library size (sum of length-normalized counts), shape `(cells, 1)` |

### Length normalization

The `length_norm` layer is computed as:

```
length_norm = (raw_counts / transcript_length) × median(transcript_length across all genes)
```

This rescales expression to be in comparable units across genes while preserving raw-count scale for the negative binomial likelihood.

---

## Splicing modality — `mdata["splicing"]`

The splicing AnnData follows the [SplicingDataset format](https://daklab.github.io/LeafletFA/data-layer/splicingdataset/) defined by LeafletFA. The required raw layers are:

| Layer | Shape | Contents |
|-------|-------|----------|
| `cell_by_junction_matrix` | `cells × junctions` | Integer junction read counts (sparse) |
| `cell_by_cluster_matrix` | `cells × junctions` | Total read counts across all junctions in the same ATSE event (sparse); the denominator for PSI |

From these, three additional layers are computed during MuData assembly:

| Layer | Shape | Contents |
|-------|-------|----------|
| `junc_ratio` | `cells × junctions` | PSI values: `junction_counts / atse_total_counts`, or 0 where unobserved |
| `psi_mask` | `cells × junctions` | Binary mask: 1 where `atse_total_counts > 0` (junction is observed in that cell), 0 otherwise |

### What is an ATSE?

An **Alternative Transcript Structure Event (ATSE)** is a group of junctions that compete within a single splicing choice (e.g. a cassette exon skip has one skipping junction and two inclusion junctions). All junctions belonging to the same ATSE share the same `cell_by_cluster_matrix` denominator value within a cell.

### PSI mask

The `psi_mask` is critical to correct model behavior. SpliceVI's partial encoder processes **only observed junctions** (mask = 1) per cell, completely ignoring positions where `psi_mask = 0`. This is what allows the model to handle the 70–95% sparsity typical in single-cell splicing data without any imputation at the input level. 


### var metadata

| Column | Contents |
|--------|----------|
| `junction_id` | Unique junction identifier (e.g. `chr:start:end:strand`) |
| `event_id` | ATSE group identifier (shared across all junctions in the same event) |
| `modality` | `"Splicing"` |

---

## Shared obs metadata

`mdata.obs` is the union of obs columns from both modalities. Typical columns include:

| Column | Description |
|--------|-------------|
| `cell_type` / `broad_cell_type` | Cell type annotation |
| `tissue` | Tissue of origin |
| `age` / `age_numeric` | Age (string or parsed integer) |
| `mouse.id` / `donor_id` | Batch/donor identifier used for batch correction |


---

## Passing the MuData to SpliceVI

```python
import mudata
from splicevi import SPLICEVI

mdata = mudata.read_h5mu("train_70_30_combined.h5mu")

SPLICEVI.setup_mudata(
    mdata,
    rna_layer="length_norm",
    batch_key="mouse.id",
)

model = SPLICEVI(mdata, n_latent=30)
```

See [Getting Started](getting-started.md) for a full training example.
