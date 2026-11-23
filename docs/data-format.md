# SpliceVI MuData Object

SpliceVI takes a [MuData](https://mudata.readthedocs.io/) object as input, with two modalities keyed `"rna"` and `"splicing"`. This page describes the required structure of each modality and how the MuData is assembled from raw inputs.

---

## Overview

```
MuData
├── mdata["rna"]       AnnData   (cells × genes)       gene expression
└── mdata["splicing"]  AnnData   (cells × junctions)   splicing
```

Both modalities share the same `obs` (cell barcodes / metadata) index. All cells must be present in both modalities.

---

## RNA modality (`mdata["rna"]`)

| Slot | Contents |
|------|----------|
| `.X` | Raw gene expression counts `(cells × genes)`, integer dtype |
| `.layers["length_norm"]` | Length-normalized expression (raw counts divided by transcript length and rescaled to the median transcript length across genes), used as the model's expression input |
| `.var["gene_id"]` | Ensembl gene ID |
| `.var["modality"]` | `"Gene_Expression"` |
| `.obsm["X_library_size"]` | Per-cell library size (sum of length-normalized counts), shape `(cells, 1)` |

### Length normalization

The `length_norm` layer is computed as follows.

```
length_norm = (raw_counts / transcript_length) × median(transcript_length across all genes)
```

This corrects for transcript length while keeping values on a count-like scale for the count likelihood (zero-inflated negative binomial by default).

---

## Splicing modality (`mdata["splicing"]`)

The splicing AnnData follows the [SplicingDataset format](https://daklab.github.io/LeafletFA/data-layer/splicingdataset/) defined by LeafletFA. It needs these two layers of read counts.

| Layer | Shape | Contents |
|-------|-------|----------|
| `cell_by_junction_matrix` | `cells × junctions` | Integer junction read counts (sparse) |
| `cell_by_cluster_matrix` | `cells × junctions` | Total read counts across all junctions in the same ATSE (sparse), the denominator of PSI |

Two more layers are computed from them when the MuData is assembled.

| Layer | Shape | Contents |
|-------|-------|----------|
| `junc_ratio` | `cells × junctions` | PSI values, `junction_counts / atse_total_counts`, or 0 where unobserved |
| `psi_mask` | `cells × junctions` | Binary mask, 1 where `atse_total_counts > 0` (the junction is observed in that cell) and 0 otherwise |

### What is an ATSE?

An **alternative transcript structure event (ATSE)** is a group of junctions in a gene that compete at shared splice sites. Two junctions belong to the same ATSE if they share a donor or an acceptor site (connected through other junctions of the ATSE). A cassette exon, for example, is one ATSE with a skipping junction and two inclusion junctions. Junctions that share no splice site with another junction are not part of any ATSE. All junctions of the same ATSE share the same `cell_by_cluster_matrix` value within a cell, and the model's junction usages within an ATSE sum to 1.

### PSI mask

The `psi_mask` tells the model which values are observed. SpliceVI's partial encoder processes **only observed junctions** (mask = 1) in each cell and ignores positions where `psi_mask = 0`, and the splicing likelihood is evaluated on observed ATSEs only. Unobserved junctions are therefore treated as missing, not as zero usage. 


### var metadata

| Column | Contents |
|--------|----------|
| `junction_id` | Unique junction identifier, `chrom_start_end_strand` (e.g. `chr1_52199887_52207300_-`) |
| `event_id` | ATSE identifier (shared by all junctions of the same ATSE) |
| `modality` | `"Splicing"` |

---

## Shared obs metadata

`mdata.obs` is the union of obs columns from both modalities. Typical columns are listed below.

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

# The layer and modality names above are the defaults, so only the batch key is given here.
SPLICEVI.setup_mudata(mdata, batch_key="mouse.id")

model = SPLICEVI(mdata)
```

See [Getting Started](getting-started.md) for a full training example.
