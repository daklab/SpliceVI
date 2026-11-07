# Tutorials plan

Audit of every tutorial, notebook and docs page in this repository, and a proposal for two tutorials. Audit only: nothing in the docs or code is changed here.

Audited: `origin/main` at `651e44b` (7 Nov 2026). The paper recipe is `SPLICEVI.preset("paper")`, with κ (`psi_prior_pseudocount`) 0.

## Summary

- **There are no tutorials or notebooks.** The README lists `tutorial.ipynb` as a future addition.
  - The material for users is the README, 8 docs pages (mkdocs, published at daklab.github.io/SpliceVI) and two Slurm job scripts.
- **No page runs as written.** Every docs example calls `SPLICEVI.setup_mudata` without `modalities=`, and main raises `ValueError: Modalities cannot be None.`
  - Checked by running the docs call on main against a toy MuData.
- **No page uses the paper presets.** The presets (`preset`, `filter_read_support`) were merged in `ea820d9` and are not mentioned anywhere in the docs.
- **No page downloads data.** Examples read a local `train_data.h5mu` or cluster paths under `/gpfs/commons/...`. No public data or checkpoint exists yet; the Zenodo records are planned for submission (`SpliceVI-utils/manuscript/paper/ZENODO_MANIFEST.md`).
- **The "sparse" preset on main still sets κ 0.1.** This is settled, not open: the 6 Nov decision takes κ out of both presets.
  - The change is commit `93bcee5` on `kisaev/paper-presets` (PR #3), not yet merged into main. Once PR #3 merges, both presets have κ 0 and "sparse" equals "paper" unless `joint=True`.
  - The tutorials below use `preset("paper")`, and κ 0 throughout.

## Audit

"Runs" means it runs on main as written. "Paper preset" means it uses `preset("paper")` or reproduces its settings, with κ 0.

| Page | Runs on main | Paper preset (κ 0) | Data it downloads | What is outdated |
|---|---|---|---|---|
| `README.md` | Scripts only (see the two job scripts below) | No: points to the job scripts, whose defaults are not the paper recipe. κ is 0 by default | None; cluster paths in the scripts | No mention of presets, DS / DE options or read support. "Future additions: tutorial.ipynb, Hugging Face models" still open. Check the author name in the citation (`Isaev, Keren`) |
| `docs/index.md` (minimal example) | No: `setup_mudata` without `modalities=` raises | No: class defaults (κ 0, but `psi_input="raw"`, Jeffreys penalty, no decoder intercept or depth covariates) | None; reads `train_data.h5mu` | Links `models/splicevi.md`, which does not exist. Ecosystem table points to ATSEmapper and LeafletFA (links not checked here) |
| `docs/getting-started.md` | No: same `setup_mudata` error | No: sets non-paper arguments (`max_epochs=800`, KL warmup 200, `rna_layer=None` instead of `"length_norm"`) | None | The "Data format" section names the modality `splice` and layers `junc_counts`, `atse_counts`, `psi_observed_mask`. The code uses `splicing`, `cell_by_junction_matrix`, `cell_by_cluster_matrix` and `psi_mask`. It contradicts `data-format.md` |
| `docs/data-format.md` | No: same `setup_mudata` error in its last example | Not applicable (data structure) | None | Mostly correct. Says "three additional layers are computed" and lists two. Does not say how to build the MuData (no builder in this repo; the paper's builder is in SpliceVI-utils) |
| `docs/outputs/latent.md` | Not as written: depends on a set-up model. `sc.pp.neighbors(mdata, ...)` on a MuData object is untested | Not applicable | None | API calls match main |
| `docs/outputs/imputed.md` | Not as written (needs a model) | Partly: describes the DM posterior mean, which is not what the paper's DS uses | None | Missing `get_normalized_splicing_DM(sample=True)` (the posterior sample used by the paper's DS) and `get_normalized_expression_ZINB` |
| `docs/outputs/differential-splicing.md` | Not as written (needs a model) | No: recommends `norm_splicing_function="dm_posterior_mean"`. The paper uses `dm_posterior_sample` with δ 0.1, FDR 0.05, then `filter_read_support(min_units=3)`. DE example uses the decoder default; the paper uses `zinb_expected` | None | No read support. Says to "consider filtering" sparse junctions but gives no rule |
| `docs/api/splicevi.md`, `docs/api/splicevae.md` | Built by mkdocstrings from docstrings | Not applicable | None | `setup_mudata` docstring example calls `scvi.model.SPLICEVI` and passes `modalities` in a form that differs from `train_splicevi.py`. `presets.py` has no API page |
| `mkdocs.yml` | Nav lists `models/splicevi.md` and `models/eddisplice.md`, which do not exist | Not applicable | None | Two missing nav pages. No page for presets |
| `train_splicevi.sh` (job script) | Only on our cluster: hard-coded `/gpfs` paths and Smriti's conda | No: defaults are an earlier recipe (`per_dimension_weighted_average`, Jeffreys, `PHI_INIT=log100`, 800 epochs; no `psi_input`, decoder intercept or depth covariates). κ 0 by default. No `--preset` flag | None; reads the October 2025 combined train file | Training file is the 2025 combined TMS + Allen file, not the 2026-06-17 per-atlas files |
| `eval_splicevi.sh` (job script) | No: passes `--mean_bayes_impute` and `--mean_bayes_group_by`, which `eval_splicevi.py` no longer accepts after `651e44b` (argparse error) | Not applicable | None; reads 2025 files and May 2026 model folders | MeanBayes flags, model folder list and test files are out of date |

Not audited: the `tests/` folder (unit tests, not user material), and the deployed site, assumed to match `docs/`.

## Proposal A: "Reproduce the paper" tutorial

**Who it is for:** a reader of the paper who wants to check our numbers and see what SpliceVI outputs, without training.

**Form:** one notebook, `tutorials/reproduce_paper.ipynb`, rendered into the docs. It runs on a laptop CPU in under 15 minutes.

**Data (downloaded in the notebook from Zenodo, after submission):**
- the Allen SpliceVI rep 1 checkpoint (in Zenodo record 1, about 0.1 GB);
- a tutorial subset of the Allen held-out test cells, for example 3,000 cells across the cell groups of Figure 2a. It has to be made and added to Zenodo; the full test file is 21.7 GB, too large for a tutorial;
- optionally, the TMS checkpoint and a TMS subset, for a second atlas.

**Steps:**
1. Load the checkpoint and the subset. Show `preset("paper", atlas="allen")` and check that the checkpoint's settings equal it (as `tests/test_presets.py` does). Show that κ is 0.
2. **Per-cell ψ:**
   - decoded ψ from the latent (`get_normalized_splicing`);
   - the DM posterior mean, which also uses the cell's own reads (`get_normalized_splicing_DM`).
   - Say when to use each.
3. **Per-cell-type ψ:** the mean of decoded ψ over the cells of each type, next to the observed pooled ψ (sum of junction reads over sum of event reads in those cells).
4. **Plots:**
   - Gria2 flip/flop and Nrxn1 SS4 across Allen cell groups, predicted against observed. A small version of Figure 2a, on one donor split.
   - A UMAP of the joint latent colored by per-cell ψ of one junction.
   - Per-cell ψ against read depth for one event, to show shrinkage at low coverage.
5. **Differential splicing with read support:**
   - `model.differential_splicing(..., **cfg["ds"])` (DM posterior sample, δ 0.1, FDR 0.05), then `filter_read_support(ds, **cfg["ds_filter"])` (at least 3 cells with reads in each group).
   - Plot effect size against `proba_ds`, with calls marked.
   - Compare `effect_size` with `emp_effect` for the supported calls.
6. Differential expression with `cfg["de"]` (`zinb_expected`), in a short cell.
7. Point to `SpliceVI-utils/manuscript/paper/` for rebuilding every figure, and to the source data on Zenodo.

**Checks it carries:** state the expected numbers on the subset (computed once when the subset is made), so a reader sees whether they reproduced them.

## Proposal B: "Use SpliceVI on your data" tutorial

**Who it is for:** a user with their own single-cell data (Smart-seq-like, or a sparser 3′ protocol) who wants per-cell and per-cell-type splicing and differential splicing.

**Form:** one notebook, `tutorials/your_data.ipynb`, with a GPU recommended for training.

**Data:** a small public example (the same Allen tutorial subset as A, treated as "your data") so that the notebook runs end to end. Each step names what to replace with the user's own files.

**Steps:**
1. **Inputs:**
   - a SplicingDataset h5ad from ATSEmapper (`cell_by_junction_matrix`, `cell_by_cluster_matrix`, `event_id`);
   - an expression h5ad with raw counts.
2. **Build the MuData:**
   - a `length_norm` layer, `junc_ratio` and `psi_mask`, and `obsm["X_library_size"]`;
   - matching cell order, with a train / test split by cell, stratified by cell type.
   - This needs a small builder function in the package, or a cell that does it (see Decisions). Show the checks: masks, event totals equal across an event's junctions, no cell without expression.
3. **Set up and train with the paper preset:**
   - `cfg = preset("paper", atlas=...)`, `setup_mudata(..., modalities={"rna_layer": "rna", "junc_ratio_layer": "splicing"}, **cfg["setup"])`, `SPLICEVI(mdata, **cfg["init"])`, `train(**cfg["train"])`.
   - Explain the `atlas` choice: no batch key, or a batch key with the adversarial term.
   - κ stays 0. Training curves and early stopping.
4. **Check the model on held-out cells:**
   - kNN cell-type accuracy of the joint latent;
   - decoded ψ against observed ψ on held-out cells, pooled by cell type.
5. **Per-cell and per-cell-type ψ, and plots:** as in Proposal A steps 2 to 4, written as functions the user can call on any event or gene name.
6. **Differential splicing with read support:**
   - between two cell types, and one-vs-rest over all types;
   - with `filter_read_support(min_units=3)`;
   - how to read `n_obs_group1/2`, and why the filter comes after the FDR call.
   - Export a results table per comparison.
7. **Sparse protocols:**
   - what changes for nuclei or 3′ data (metacells, read support counted in metacells);
   - κ 0, as in both presets after PR #3; `preset("sparse", joint=True)` for a joint reference + sparse model (batch key `technology`, dispersion `gene-batch`).

## Docs fixes that both tutorials need (separate PR, after approval)

- Add `modalities=` to every `setup_mudata` example.
- Make the layer and modality names the same across pages, matching the code.
- Add a "Presets" page (`preset`, `filter_read_support`) and an API entry for `presets.py`.
- Rewrite the DS page around the paper settings (DM posterior sample, read support). Keep the posterior mean as an option, with what it does.
- Remove the missing `models/` pages from the nav, or write them.
- Fix `eval_splicevi.sh` (MeanBayes flags), and point `train_splicevi.sh` at the presets or label it an internal example.

## Decisions

Settled: κ 0 in both presets (6 Nov decision; `93bcee5`, PR #3). The tutorials need PR #3 merged before they mention the "sparse" preset.

1. **Tutorial subset:** which cells (proposal: 3,000 Allen test cells from the Figure 2a cell groups), and add it to the Zenodo record 1 upload list.
2. **MuData builder:** add a small `splicevi.build_mudata()` (moved from the SpliceVI-utils data stage), or keep it as notebook code.
3. **Where tutorials live:** `tutorials/` in this repository and rendered in mkdocs (mkdocs-jupyter), or docs pages only.
4. **When:** before submission (30 Nov), or after the Zenodo records exist. Proposal A needs the Zenodo DOIs to download from.
