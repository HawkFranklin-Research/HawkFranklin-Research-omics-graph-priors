# BRCA v2 pre-specified protocol (E5 and E6: replication of H1–H4 in breast cancer)

| | |
|---|---|
| Written | 2026-10-31, by the scientific guide, at the author's request |
| Status | **Frozen 2026-10-31**, approved by the author and merged into `v2-pipeline` before any BRCA model run. Only fields marked `<to record>` (data revisions, label sources) may be filled in later. Any other change is a deviation. Smoke and timing runs never change this protocol |
| Approved choices | LumA vs LumB AUROC is the primary classification endpoint, with 4-class PAM50 as pre-specified secondary. PFS is a pre-specified secondary survival endpoint. Part B is deferred to the revision (R1) |
| Based on | `configs/v2/protocol_hnscc.md` (frozen) and the E1 design. Every difference from HNSCC is listed in §16 with its reason |
| Changes after freezing | Recorded as dated deviations in `STUDY_DESIGN.md` §10 and in the paper |
| Who does what | The guide writes this protocol. The coding agent implements §17, the testing agent runs the smoke and timing checks on small data, and the execution agent runs the full grid on gcloud (§15). Roles as in `AGENTS.md` |

**Structure.** The protocol has two parts, covering up to three BRCA datasets:

| Part | Experiment | Dataset | Role | Design |
|---|---|---|---|---|
| **A** | E5 | TCGA-BRCA | **Primary**: replication of H1–H4 | Identical to HNSCC E1: within-cohort repeated CV, same 8 comparisons, same statistics |
| **B** | E6 | CPTAC-BRCA | Secondary: external test set | The TCGA-trained models are applied unchanged; no training on this cohort |
| **B** | E6 | a third BRCA cohort (likely METABRIC) | Secondary: external test set | As for CPTAC-BRCA, once its eligibility (§14.2) is confirmed |

**Author decision, 2026-10-31:** Part A runs now and goes into the first submission (M1). Part B is deferred to the revision (R1), so that missing methylation, label curation and microarray platform shifts cannot delay submission. Part B is specified now (§14) so that it is fixed before any external data is modelled.

**What was looked at before writing this.** Only label-blind and descriptive data: label counts, event counts, covariate completeness, and a preview of which genes the node rule selects (§3). No model was run on BRCA, and no BRCA outcome was related to any feature.

---

## 1. Purpose

E5 tests whether the HNSCC findings (E1) hold in a second cancer with a different biology and a larger cohort. It uses the same four hypotheses, the same 8 comparisons, the same models, hyperparameters and statistics.

Sections 2–13 define Part A. Section 14 defines Part B.

Part A also adds one thing HNSCC could not test. HPV status was too easy: every model, including the untrained one, scored AUROC ≥ 0.94, so "the downstream endpoint does not depend on the encoder" might only be a ceiling effect. BRCA's primary classification endpoint is chosen to be hard but well powered, **luminal A vs luminal B**, so the downstream question is asked again without a ceiling.

## 2. Cohort and data

| Item | Specification |
|---|---|
| Cohort | TCGA-BRCA (PanCancer Atlas), primary tumours, one sample per patient |
| Inclusion | Every patient with an RNA-seq profile passing the existing cube rule (`cube.required_channels: [rna]`, `min_rna_coverage: 0.9`). Expected: about 1,082 patients |
| Other channels | Methylation (1,066), CNV log2 (1,070), somatic mutation (1,009). All 4 channels: 978. A missing channel is masked, as in HNSCC; no patient is dropped for it |
| Sex | 1,070 female and 12 male. Male patients are kept and disclosed; sex is not a covariate |
| Exclusions | None beyond the cube rule. CPTAC-BRCA and the third cohort are never used in Part A: not for nodes, graphs, normalization, training or model selection (Part B, §14) |
| Data files | `harmonized/tcga/BRCA/{rnaseq_gene,methylation_gene,cnv_log2,mutation_binary}` (`.parquet`, `.samples.parquet`, `.features.parquet`) and `clinical/{patient_master,targets}.parquet` |
| Data revision | The BRCA tables are uploaded to the HF dataset in one commit before the full run. That commit hash is recorded here, `a4ea22bafc5a838f550589150670859b1a5b170f`, and in the run manifest |

## 3. Channels, nodes and graphs: identical to HNSCC

- **Channels:** RNA (log2), methylation (β → M-value), CNV log2 (clipped), mutation binary. Same transforms and HGNC harmonization.
- **Node rule:** the MSigDB Hallmark union, then the top 2,000 genes by RNA variance over all included patients. The rule is label-blind, and the choice of N = 2,000 is unchanged.
- **Graphs:**
  - STRING v12 with combined score ≥ 700
  - co-expression kNN built per fold from training patients only, trimmed to the STRING edge count
  - degree-preserving rewiring of each
  
  Same seeds and swap count.
- **Disclosed in the paper:**
  - the Hallmark composition of the node set (`03_nodes/hallmark_composition.csv`)
  - the number of STRING-isolated nodes and the degree quantiles (`04_graphs/string/*/diagnostics.json`)

**Label-blind preview** (2026-10-31, approximate: raw symbols and log2 values, no harmonization):
- About 4,200 Hallmark genes have RNA data.
- The marker genes needed for the controls in §10 fall inside the top 2,000. Approximate variance ranks: ESR1 31, PGR 26, GATA3 255, FOXC1 316, ERBB2 754, MKI67 1,051.
- HALLMARK_MYOGENESIS contributes about 135 genes, as in HNSCC (about 150). The muscle-gene signal is therefore not HNSCC-specific; it probably reflects stromal and smooth-muscle content. The BRCA node set is **not** altered for this. E4 decides whether a sensitivity run applies to both cancers.

If the real `03_nodes/nodes.csv` leaves out a control gene, that control is reported as "not testable". It is not replaced by another gene.

## 4. Endpoints

PAM50 labels come from `SUBTYPE`. Counts among RNA patients: LumA 499, LumB 197, Basal 171, Her2 78, Normal-like 36, missing 101.

| ID (metric name) | Definition | Patients | Role |
|---|---|---|---|
| `luma_lumb` (`luma_lumb_auroc_gene_latent_pca`) | Binary: LumB = 1, LumA = 0, and NaN for everyone else (Basal, Her2, Normal-like, missing). Only the classifier fit and evaluation use the 696 labelled patients | 696 (197 positive) | **Primary** classification endpoint |
| `os` (`os_delta_c_gene_latent_pca`) | Gain in Harrell's C on OS: (clinical + embedding) minus clinical-only Cox model, covariates in §5 | about 1,082 (151 events) | **Primary** survival endpoint |
| `node_rna_r2` | R² of RNA on genes whose whole node was masked, in test patients | about 1,082 | **Primary** reconstruction endpoint |
| `pam50` (`pam50_macro_auroc_*`, `pam50_balanced_accuracy_*`) | 4-class: LumA, LumB, Basal, Her2. Normal-like and missing excluded | 945 | Secondary. Expected near ceiling: PAM50 is defined from RNA |
| `basal` (`basal_auroc_*`) | Binary: Basal = 1 vs LumA, LumB and Her2 = 0. Normal-like and missing excluded | 945 (171 positive) | Positive-control endpoint (§10). Not a hypothesis test |
| `pfs` (`pfs_delta_c_gene_latent_pca`) | Gain in Harrell's C on PFS, same covariates and method as `os` | about 1,082 (145 events) | **Secondary** survival endpoint |

**Labels never shrink the cohort.** Any patient whose label is NaN for an endpoint stays in everything else:
- the cube
- node selection
- per-fold normalization
- co-expression graphs
- autoencoder training
- masking
- reconstruction metrics
- survival models

This is how the 28 HPV-unknown HNSCC patients were handled. Classification heads are fitted and scored only on training and test patients with a finite label. A test fold whose labelled subset lacks a class is a failed endpoint (§13).

**Why PFS is secondary.** TCGA-BRCA overall survival is about 86% censored, with short follow-up, and the TCGA-CDR resource cautions against relying on OS alone in such cohorts. PFS is added as a pre-specified secondary endpoint. It has 145 events, about the same as OS, so it adds robustness rather than power. OS stays primary for symmetry with HNSCC.

**Why LumA vs LumB is primary.** It is clinically meaningful: a continuous proliferation boundary (MKI67, AURKA, CCNB1) with prognostic weight. It is large (696 patients, about 39 LumB per test fold) and known to be imperfectly separable from bulk RNA, so it has no ceiling. It is also binary, like HPV, so the HNSCC evaluation, pooled bootstrap and scalar-logit attribution apply unchanged. The PAM50 4-class AUROC, recommended earlier as primary (`STUDY_DESIGN.md` D2), is demoted to secondary for the same reason HPV failed in HNSCC: it is RNA-defined and would saturate. This decision uses E1's HNSCC experience only, not any BRCA result.

**Readouts.** The primary readout is `gene_latent_pca` (16 components fitted on the fold's training patients), as in HNSCC. `mean_pool` is secondary.

## 5. Clinical covariates for ΔC

The clinical-only Cox model and the joint model both contain:

- `age`: z-scored with the training mean and SD
- `stage_num`: from `AJCC_PATHOLOGIC_TUMOR_STAGE`, I–IV. "Stage X" and missing (19 patients) get the training median plus a `stage_missing` indicator
- `pam50`: one-hot with fixed categories {LumB, Basal, Her2, Normal-like}, LumA as the reference, plus `pam50_unknown` for missing

**Why PAM50 is a covariate.** It is BRCA's analogue of HPV in HNSCC: the established molecular prognostic label. ΔC then asks whether the embedding adds anything beyond the known subtype.

**Power note.** 151 events across 5 folds means about 30 test events per fold. Expect wide confidence intervals. The Cox penalizer search (0.1, 1, 5) is unchanged.

## 6. Folds

- 5-fold stratified CV × 3 repeats; seeds `1000 + repeat` (outer) and `2000 + repeat` (inner); 15% of the outer training set used for validation and early stopping. All identical to HNSCC.
- **Strata:** raw `SUBTYPE` (6 levels: the 5 PAM50 classes plus missing) × `OS_event`. The smallest cell, Normal-like with an event, has 7 patients, which is at least the 5 folds required. If any cell is too small, the existing fallback order (label × event → label → event → none) applies, and the choice is written to the fold report.
- **Fold check (hard):** every outer test fold must contain at least:
  - 10 LumB and 10 LumA patients
  - 10 Basal patients
  - 10 OS events
- The fold and mask files are hashed, as in HNSCC.

## 7. Conditions and baselines

The 10 model conditions are identical to HNSCC:
- `gatv2_string` (reference), `gcn_string`, `sage_string`
- `gatv2_string_rewired`, `gatv2_coexpr`, `gcn_coexpr`, `gatv2_coexpr_rewired`
- `node_mlp`, `dense_ae`, `gatv2_string_untrained`

The baselines are `gene_mean`, `lowrank_impute`, `pca_concat`, `pca_rna`, and **`single_gene_esr1`**, which replaces `single_gene_cdkn2a` (§10).

**MOFA+:** included only if it is also added to HNSCC, with the same settings. In either cancer it is an exploratory H1 comparator, not one of the 8 primary comparisons; the frozen HNSCC protocol fixed those.

**Hyperparameters:** the values in the frozen `configs/v2/base.yaml`, copied unchanged into `configs/v2g/base.yaml`:
- hidden size 64, latent size 16, 1 head, dropout 0.1
- 60 epochs, patience 10, learning rate 1e-3, weight decay 1e-4
- 15% of entries masked and 10% of whole nodes masked

No BRCA-specific tuning.

## 8. Primary hypotheses and tests

These are the same 8 comparisons as HNSCC. Each is `gatv2_string` minus the comparator.

| | Comparators |
|---|---|
| H1 | `dense_ae`, `node_mlp`, `pca_concat`, `lowrank_impute` |
| H2 | `gatv2_string_rewired` |
| H3 | `gcn_string`, `sage_string` |
| H4 | `gatv2_coexpr` |

**Primary metrics:** `node_rna_r2`, `luma_lumb_auroc_gene_latent_pca` and `os_delta_c_gene_latent_pca`. That gives 24 primary tests.

**Test:** a Nadeau–Bengio corrected resampled t-test paired by (repeat, fold), with Holm correction across the 8 comparisons within each metric. A comparison needs at least 2 finite paired folds; otherwise it is reported as not estimable. `lowrank_impute` and `gene_mean` have no patient embedding, so they enter only the reconstruction tests.

**Known built-in asymmetries, stated in advance:**
- `node_mlp` has no input for a fully masked node, so H1 against it on NM-R² is trivially won.
- Co-expression graphs are built from RNA, so an H4 advantage on RNA is partly built in. Methylation and CNV NM-R² are the cleaner H4 evidence (exploratory).

## 9. What counts as replication (decided before the run)

For each of the 24 primary tests, the BRCA result is classified against HNSCC E1:

| BRCA outcome | Label |
|---|---|
| Same sign as HNSCC and Holm p < 0.05, where HNSCC was also significant | **replicated** |
| Same sign, Holm p ≥ 0.05, where HNSCC was significant | **not replicated (same direction)** |
| Opposite sign and Holm p < 0.05 | **reversed** |
| Holm p ≥ 0.05, where HNSCC was also p ≥ 0.05 | **consistent null** |
| Holm p < 0.05, where HNSCC was p ≥ 0.05 | **new effect** |

**Equivalence** is claimed only when the 95% CI of the paired difference lies entirely within ±0.02 (for R² and AUROC) or ±0.01 (for ΔC). Equivalence claims are exploratory.

**The paper's four headline claims** are stated as replicated in BRCA only if the following hold:

| Claim | Condition in BRCA |
|---|---|
| C1 | Low-rank imputation and PCA beat STRING-GATv2 on NM-R²(RNA): H1 reversed, replicated |
| C2 | Real STRING beats rewired STRING on NM-R²(RNA): H2, replicated |
| C3 | Co-expression GATv2 beats STRING GATv2 on NM-R²(RNA): H4, replicated |
| C4 | GATv2 does not differ from GCN or SAGE on any primary metric: H3, consistent null |

**The downstream claim** ("the clinical endpoints do not depend on the encoder") is tested on LumA vs LumB, which has no ceiling:
- **Supported** if none of the 8 comparisons on `luma_lumb_auroc` is significant **and** the untrained encoder is within 0.02 AUROC of the trained one.
- Otherwise the paper reports that the downstream insensitivity seen in HNSCC was at least partly a ceiling effect.

All 24 results are reported whatever their direction. A failure to replicate is reported as such; nothing is rerun or re-tuned to obtain replication.

## 10. Controls and hard gates

**Hard gates.** If one fails, stop, investigate as a pipeline fault, and do not report the grid until it is resolved:

| Gate | Requirement |
|---|---|
| `single_gene_esr1` on the `basal` endpoint | AUROC ≥ 0.80 in every fold. ESR1 is the canonical ER gene; basal tumours are ER-negative |
| `gene_mean` | All reconstruction R² within ±0.02 of 0 |
| Fold check | §6 |

**Expected, not gated (reported as checks):**
- `pca_rna` on `pam50` macro-AUROC > 0.90
- `gatv2_string_untrained` NM-R²(RNA) ≤ 0
- `gatv2_coexpr_rewired` below `gatv2_coexpr` on NM-R²(RNA)

**Circularity control:** same as HNSCC. KMeans cluster-recovery AUROC is compared for trained, untrained and random-Gaussian embeddings.

## 11. Interpretation (E3 analogue)

**Attention stability gate:** the median cross-repeat gene RBO for `gatv2_string` layer 1 must be ≥ 0.5, unchanged. Report it against the shuffled null and against the untrained model, with and without the degree weight. These are the HNSCC exploratory analyses, now pre-specified.

**Attribution:** on repeat 0, 40 test patients per fold, the same 5 conditions as HNSCC (`gatv2_string`, `gatv2_string_rewired`, `gatv2_coexpr`, `gatv2_coexpr_rewired`, `gatv2_string_untrained`). Knockout and gradient×input are computed for 3 targets:

- the embedding
- the **`basal` logit** (positive-control target)
- the **`luma_lumb` logit** (primary-endpoint target)

**Positive-control expectations** (pre-specified). They validate the attribution method, not the graph:

| Target | Gene rank (knockout) | Hallmark prerank (FDR < 0.25 in ≥ 3 of 5 folds) |
|---|---|---|
| `basal` logit | ESR1 at the ≥ 95th percentile in `gatv2_string` | HALLMARK_ESTROGEN_RESPONSE_EARLY or _LATE |
| `luma_lumb` logit | MKI67 at the ≥ 90th percentile in `gatv2_string` | HALLMARK_E2F_TARGETS or HALLMARK_G2M_CHECKPOINT |

**Graph specificity** (the lesson from HNSCC). Every control rank and enrichment is reported for all 5 attribution conditions. A gene or pathway is described as graph-dependent only if `gatv2_string` exceeds **both** `gatv2_string_untrained` and `gatv2_string_rewired` in at least 4 of 5 folds. Otherwise it is attributed to the classifier and the data.

**Enrichment:**
- **Always run:** on the two logit targets (`basal`, `luma_lumb`), as positive controls.
- **Only if the attention gate passes:** attention-score and embedding-knockout enrichment.

The settings are as in HNSCC: gseapy 1.3.1, the node universe as background, 1,000 permutations, seed 0.

## 12. Secondary and exploratory analyses

All of these are labelled exploratory in the paper and have no multiplicity claim. They use uncorrected paired tests against the reference, as in HNSCC.

1. `pam50` macro-AUROC and balanced accuracy; `basal` AUROC.
2. Uno's C for OS and PFS; `mean_pool` versions of every downstream metric. PFS ΔC is a pre-specified secondary endpoint (§4) and is reported with the secondary results.
3. Entry-masked (EM) R² and NM-R² for methylation, CNV and mutation (AUROC).
4. NM-R² by STRING degree: isolated, 1–5, 6–25, >25. Same bins and code as HNSCC (`manuscript/figure_code/analysis_degree_recon.py`).
5. Attention overlap with and without the degree weight (`analysis_attention_alpha.py`).
6. A cross-cancer concordance table: for every primary test, HNSCC Δ and BRCA Δ side by side with their labels from §9.

## 13. Missing data and failures

- **A failed fit** (for example, the loss-improvement check) is kept in `missing_runs.csv`, reported, and not rerun.
- **A failed endpoint in a fold** (for example, a class absent from the test fold, or a Cox fit failure at all three penalties) goes to `endpoint_failures.csv`. That fold is then missing for that comparison.
- **If more than 2 of the 15 fits of any condition fail**, the condition's results are reported but flagged as unreliable.

## 14. Part B: external BRCA cohorts (E6)

### 14.1 Principle

Part B asks one question: do the TCGA-BRCA ranking of conditions and the headline claims C1–C4 hold when the trained models meet patients from another cohort and platform?

- **No retraining.** Each of the 15 Part A fold models per condition (10 conditions, plus baselines refit as below) is applied unchanged to the whole external cohort.
- **External cohorts never inform any choice:** no nodes, graphs, hyperparameters, thresholds or condition selection.
- **Part B results are secondary.** They support or qualify Part A; they never override it.

### 14.2 Cohorts and eligibility

| Cohort | Status | Channels available | Labels | Survival |
|---|---|---|---|---|
| CPTAC-BRCA | In `processed/harmonized/cptac/BRCA/` | RNA (121), CNV log2 (113), mutation (122); **no methylation** | **None in our tables.** PAM50 calls must be added from the source publication (Krug et al., 2020) before Part B starts | 2 OS events: **no survival analysis** |
| Third cohort (candidate: METABRIC) | Not downloaded | Expected: microarray RNA, CNA, a 173-gene mutation panel; no methylation | PAM50 provided with the data | About 1,900 patients with long follow-up |

A cohort enters Part B only if all of these hold:
- at least 100 patients with RNA
- at least 90% of the 2,000 Part A RNA nodes observed (after HGNC harmonization)
- PAM50 labels available from the data source or its publication, never computed by us from RNA

A cohort that fails is reported as excluded, with the reason.

Labels are attached before any model is applied, and the label source is recorded in this file as an amendment before the Part B run: `<CPTAC label source: to record>`, `<third cohort: to record>`.

### 14.3 Preprocessing of external cohorts

| Item | Rule |
|---|---|
| Nodes | The Part A node set (TCGA `03_nodes/nodes.csv`). A node absent from the external cohort is masked in every channel |
| Missing channel (e.g. methylation) | Masked for every patient. The models were not trained with whole-channel dropout, so this is out of distribution; that is a stated limitation. A channel-dropout condition (D5) is **not** part of E6 |
| Normalization: primary | Per-cohort, per-gene standardization using the external cohort's own mean and SD over all its patients (label-blind; disclosed as transductive) |
| Normalization: sensitivity | The TCGA fold's training-set statistics applied unchanged |
| Not used | ComBat or any batch correction that uses TCGA and the external cohort together |
| Graphs | The fold's TCGA graphs, unchanged: STRING, the TCGA-train co-expression graph, and the rewired graphs |
| Platform note | For a microarray cohort, RNA is used as provided (log2 intensities) and is then standardized per cohort; no cross-platform mapping is attempted |

### 14.4 External endpoints

| Endpoint | How | CPTAC-BRCA | Third cohort |
|---|---|---|---|
| NM-R² for each observed channel | Same masking scheme and seeds as Part A, applied to external patients | yes | yes |
| `luma_lumb`, `basal` AUROC; `pam50` macro-AUROC | The fold's TCGA readout PCA and logistic heads (fitted on TCGA training patients) applied to the external embeddings | if labels are attached | yes |
| OS ΔC | A TCGA-fitted Cox model applied to external patients, against TCGA-fitted clinical-only Cox | no (2 events) | yes, if age and stage are available |

Baselines (`pca_concat`, `pca_rna`, `lowrank_impute`, `gene_mean`, `single_gene_esr1`) are applied the same way: fitted on the TCGA fold's training patients, then applied to the external cohort.

### 14.5 Statistics and reporting

- The 15 fold models give 15 external evaluations per condition on one fixed external test set. Training sets overlap and the test set is fixed, so the Nadeau–Bengio correction does not apply.
- **What is reported:**
  - the mean and SD across the 15 fold models
  - for repeat 0 averaged over its 5 folds, a patient-level bootstrap 95% CI (1,000 resamples)
- **Comparisons:** paired differences (reference minus comparator) for the same 8 comparisons, with bootstrap CIs. **No p-values and no Holm correction.** Part B is descriptive.
- **Transfer gap:** for each condition and metric, external minus Part A within-TCGA.
- **Claim check:** each of C1–C4 is marked "holds externally" if the external paired difference has the same sign as Part A and its bootstrap CI excludes 0; otherwise "does not hold externally". For C4 (a null), it holds if the CI includes 0.
- Normalization sensitivity (§14.3) is reported for the reference and the H1 comparators only.

### 14.6 Order

Part B runs after Part A is complete. Its inputs must be on the HF dataset at a pinned revision, with labels attached and recorded in §14.2. It uses the stored Part A fold models and adds no training, so its compute is small.

## 15. Execution and provenance

1. **Order (Part A):**

   | Step | Agent | What |
   |---|---|---|
   | 1 | coding | Implement §17. Code only; nothing is run |
   | 2 | testing | pytest; the HNSCC equivalence check (§17 item 9); BRCA smoke on small data (`configs/v2g/brca_smoke.yaml`); BRCA single-fold timing. Stops at the first failure and writes `revise-2026/TEST_ERRORS.md`. The author relays errors to the coding agent until everything passes |
   | 3 | data | Upload the TCGA-BRCA tables to HF in one commit; record the revision in §2 |
   | 4 | author | Commit this protocol (the freeze) |
   | 5 | execution | Run the full grid on a gcloud T4 Spot instance from a **clean** committed tree, using the pinned HF revision |
   | 6 | guide | Check `07_eval/comparisons.csv` against §9 before any result goes into `EXPERIMENTS.md` or the paper |

2. **Order (Part B, R1 only):** after M1 is submitted, the same cycle applies. Coding implements §17 items 11–13; testing runs an external smoke on a small external subset; then the labels and external tables are uploaded and the §14.2 amendments recorded; then execution.
3. **Manifest:** must show `git_dirty: false`, the code commit, the config hash, the HF revision, and the sha256 of this protocol file. The cloud runner refuses to start on a dirty tree.
4. **Compute estimate:** E1 took 2 h 07 min on a T4 for 515 patients. BRCA has about 2.1× the patients and one extra attribution target, so expect about 5–6 h on a T4 Spot instance for Part A; Part B is well under an hour. Record both in `SPENDS.md`.
5. **Outputs:**
   - Part A: `outputs/v2/brca_v2_e5/`
   - Part B: `outputs/v2/brca_v2_e6_<cohort>/`
   
   Record the status, evidence and results in `EXPERIMENTS.md` E5 and E6.

## 16. Differences from the HNSCC protocol

| Item | HNSCC (E1) | BRCA (E5) | Reason |
|---|---|---|---|
| Primary classification endpoint | HPV AUROC | LumA vs LumB AUROC | HPV saturated; LumA vs LumB has no ceiling and is well powered |
| Clinical covariates | age, sex, stage, HPV | age, stage, PAM50 | Sex is nearly constant in BRCA; PAM50 is the subtype analogue of HPV |
| Stratification label | HPV | PAM50 (`SUBTYPE`, 6 levels) | The cancer-specific primary label |
| Single-gene positive control | CDKN2A → HPV, gate 0.75 | ESR1 → basal, gate 0.80 | The canonical marker for the cancer |
| Attribution logit targets | HPV | basal (control) and LumA vs LumB (primary) | A control target plus the primary endpoint |
| Attention-alpha and degree-stratified analyses | added after the fact, exploratory | pre-specified, still exploratory | Learned from E1 |
| External cohorts | CPTAC-HNSCC excluded (too few patients) | CPTAC-BRCA and a third cohort as external test sets (Part B) | BRCA has usable external cohorts; HNSCC does not |
| Everything else | — | identical | — |

## 17. Implementation requirements (for the coding agent; no scientific change)

These say *what* must be configurable or added. *How* is up to the coding agent.

**The HNSCC code is frozen and is not touched** (amendment A2, author decision 2026-10-31). E1–E3 were produced by commit `df42798`, tagged `hnscc-e1-frozen`. Nothing it uses may be edited:
- `graphr1/`
- `scripts/run_v2.py`, `run_v2_bounded.sh`, `run_v2_gcloud.sh`, `idle_watchdog.sh`, `install_idle_watchdog_service.sh`, `provision_gcloud_spot.sh`
- `configs/v2/*.yaml`, `configs/v2/protocol_hnscc.md`
- `tests/test_v2_primitives.py`
- `requirements-v2.txt`

**BRCA runs on a generalized copy** of that code:

| HNSCC (frozen) | BRCA copy (generalized) |
|---|---|
| `graphr1/` | `graphr1g/`: package `graphr1` copied at `hnscc-e1-frozen`; internal `graphr1.` imports become `graphr1g.` |
| `scripts/run_v2.py`, `scripts/run_v2_bounded.sh` | `scripts/run_v2g.py`, `scripts/run_v2g_bounded.sh` |
| `scripts/run_v2_gcloud.sh` | `scripts/run_v2g_gcloud.sh`. It may call the unchanged `idle_watchdog.sh` and `provision_gcloud_spot.sh` |
| `configs/v2/base.yaml` | `configs/v2g/base.yaml`, plus `brca_smoke.yaml`, `brca_gcloud_probe.yaml`, `brca_gcloud.yaml` and `hnscc_equiv_smoke.yaml` |
| `tests/test_v2_primitives.py` | `tests/test_v2g_*.py` |
| `requirements-v2.txt` | used as is; any new dependency goes in `requirements-v2g.txt` |

The copy changes only what items 1–11 require. The models, losses, masking, node rule, graphs, normalization, statistics and hyperparameters stay identical. Item 9 proves this.

In items 1–11 below, a path such as `graphr1/x.py` means its copy `graphr1g/x.py`, and `configs/v2/` means `configs/v2g/`. The originals are never edited.

**Part A (items 1–10):**

1. **Primary metrics:** `aggregate.py` hard-codes `PRIMARY_METRICS`. Read them from a per-cancer `protocol.<cancer>.primary_metrics` list in the config. The same list drives the primary-endpoint panels in `evaluate/figures.py` (currently HPV-specific) and the cohort note in the `report` stage (`pipeline.py`, currently HNSCC text).
2. **Fold check:** `pipeline.py:113` asserts `test[stratify_label].eq(1)`, which assumes a binary 0/1 label. Replace it with a configurable fold-check list (column, value, minimum count, relaxed minimum). The HNSCC-equivalent config uses `hpv = 1 ≥ 3` and `OS_event = 1 ≥ 10`; BRCA's comes from §6.
3. **Covariates:** `clinical.py` `encode_covariates` special-cases `hpv`. Add a generic categorical covariate with fixed categories, a reference level and an unknown indicator, used for `pam50`. The `hpv` encoding must stay identical in the copy.
4. **Single-gene baseline:** `single_gene_cdkn2a` is hard-coded in `stage_evaluation.py` (lines ~36, ~56 and the gate at ~170) and in `audit.py`. Generalize it to `single_gene: {gene, endpoint, min_auroc}`, with the baseline named `single_gene_<gene lower-case>`. HNSCC-equivalent: `{CDKN2A, hpv, 0.75}`. BRCA: `{ESR1, basal, 0.80}`.
5. **Attribution targets:** `stage_interpret.py` uses the first binary endpoint (`hpv`) and writes `cdkn2a_rank_percentile.csv`. Make the targets a list (`attribution_endpoints`) and the control genes a mapping (`control_genes: {endpoint: [genes]}`), writing `knockout_<endpoint>_logit.csv`, `gradxinput_<endpoint>_logit.csv` and `<gene>_rank_percentile_<endpoint>.csv`. Enrichment always runs on every configured logit target, which replaces the `"hpv_logit" in name` check.
6. **Endpoints:** in the BRCA block of `configs/v2g/base.yaml`:
   - add `binary: {luma_lumb, basal}` as defined in §4. Every non-listed `SUBTYPE` must map to NaN, which the existing `build_patient_table` mapping already does.
   - add `survival.pfs: {time: PFS_days, event: PFS_event}` next to `os` (primary).
   - keep `multiclass: pam50`.
7. **New configs.** One generalized codebase runs in three settings, and only the config differs (amendment A1):
   - `configs/v2g/brca_smoke.yaml`: a local smoke run on the laptop's processed data, small and relaxed.
   - `configs/v2g/brca_gcloud_probe.yaml`: a tiny GPU probe on the VM, reading the HF data.
   - `configs/v2g/brca_gcloud.yaml`, the full run, with:
     - `cancers: [BRCA]`
     - `experiment_name: brca_v2_e5`
     - `cv.stratify_label: SUBTYPE`
     - the BRCA fold checks
     - the HF data block (repo, pinned revision, local directory, download patterns)
     - `threads: 6`
8. **Cloud runner and provenance** (amendment A1 refines this item):
   - The new `scripts/run_v2g_gcloud.sh` takes the HF repo, revision, local data directory, download patterns and probe config **from the run config**, not from hard-coded HNSCC values. The frozen `scripts/run_v2_gcloud.sh` stays as it is, for HNSCC.
   - It refuses to start if the HF revision is still a placeholder.
   - It refuses to start if **tracked** files differ from the commit (`git status --porcelain --untracked-files=no` is not empty). Untracked data, logs and outputs do not block the run.
   - The run manifest records:
     - the commit
     - `git_dirty_tracked` and the number of untracked files, kept separately. The E1 manifest's `git_dirty: true` counted untracked files too, so it could not tell code changes from downloaded data.
     - the HF repo and revision
     - the sha256 of the cancer's protocol file
   - A `source_diff.patch` (`git diff HEAD`) is written into the output folder whenever tracked files differ.
   - The results archive must include every run's `model.pt` and its embeddings (needed by item 11).
9. **Equivalence requirement** (amendment A2 replaces the earlier regression wording). It proves the BRCA code is the same science as the frozen HNSCC code. The generalized copy, run with `configs/v2g/hnscc_equiv_smoke.yaml` (the HNSCC smoke settings expressed in the new config format), must reproduce the frozen HNSCC smoke output `outputs/v2/hnscc_v2_smoke/`:
   - identical `02_folds` and mask hashes
   - an identical node list
   - identical baseline metrics
   - model metrics identical when run on the same device, otherwise within 1e-4
   
   The testing agent runs this using the comparison script `scripts/compare_v2_runs.py`, which the coding agent writes. The frozen HNSCC code is not rerun or edited for this.

   **Exception for rank-based metrics** (amendment A3; post hoc, guide, 2026-10-31). An AUROC or AUPRC model metric that differs by more than 1e-4 still counts as equivalent only if a CPU checkpoint replay shows all of:
   - (i) identical inputs and masks
   - (ii) checkpoint parameters within 1e-5 of each other
   - (iii) prediction differences of at most 1e-6
   - (iv) the metric difference is explained by numerical precision: either the two replays give the same metric, or every reordered positive/negative pair has a score margin of at most 1e-6
   
   Everything else is unchanged. Folds, masks, nodes and baselines must match exactly. Every other model metric must agree within 1e-4. A metric that fails (i)–(iv) is a real failure.
10. **Tests:** unit tests for the generic categorical covariate, the configurable fold checks, and the single-gene baseline naming.

**Item 11 is needed now, in Part A, so that R1 can reuse the models. Items 12–13 are R1 only (E6):**

11. **Keep the Part A models.** The Part A run must keep, for every condition and fold:
    - the trained weights
    - the training-patient embeddings
    
    so that readout PCA, logistic heads and Cox models can be refitted deterministically on the TCGA training patients, with the same code as Part A. The trained weights must also go into the results archive, not only stay on the VM disk.
12. **An `external` stage.** It should:
    1. load an external cohort through the same cube and harmonization path, with whole-channel absence allowed
    2. restrict it to the Part A nodes and mask what is missing
    3. normalize it with either the primary or the sensitivity rule from §14.3
    4. apply every Part A fold model and baseline
    5. compute the §14.4 endpoints
    6. write `per_model_external_metrics.csv` and the bootstrap summaries from §14.5
    
    It must never write into the Part A output folder.
13. **External config** (`configs/v2/brca_external_<cohort>.yaml`): it points at the Part A output folder, the external data paths (HF revision), the normalization rule, and the endpoints available for that cohort.

## 18. Not part of this protocol

- Training on, or tuning with, CPTAC-BRCA or the third cohort.
- A whole-channel-dropout condition (D5), or any new model condition.
- Any change of N, of hyperparameters, or of the gene universe for BRCA.
- Using BRCA results to revise the HNSCC analysis.
- Computing PAM50 labels ourselves for any cohort.

## Amendments

| ID | Date | By | Change | Scientific impact |
|---|---|---|---|---|
| A1 | 2026-10-31 | guide | §17 implementation details: per-cancer figures and report text (item 1); three configs per cancer (smoke, probe, full) and an HF data block (item 7); a cloud runner that reads data settings from config, separates tracked from untracked dirtiness, writes `source_diff.patch` and archives model weights (item 8); a run-comparison script for the HNSCC regression (item 9) | None. §1–16 unchanged |
| A2 | 2026-10-31 | author | The HNSCC code is frozen at `df42798` (tag `hnscc-e1-frozen`) and never edited. BRCA runs on a generalized copy (`graphr1g/`, `scripts/run_v2g*.{py,sh}`, `configs/v2g/`). Item 9 becomes an equivalence check of the copy against the frozen HNSCC smoke output (§17) | None. Same methods, now in a separate copy |
| A3 | 2026-10-31 | guide (post hoc) | §17 item 9: a narrow exception for rank-based metrics (AUROC, AUPRC) whose differences are shown, by CPU checkpoint replay, to be numerical precision. The trigger was test round 3 (commit `21bc33c`; the original strict FAIL is preserved verbatim in `revise-2026/TEST_ERRORS_BRCA_round_equivalence.md`). All checks passed except 3 `node_mut_auroc` rows (gatv2_string, gatv2_string_untrained, gcn_string; repeat 0, fold 0), off by 2.8e-4 to 1.7e-3. That metric has 23 positives among 1,043 entries, so one reordered pair moves AUROC by about 4e-5. Replay showed identical inputs, checkpoints within 8.3e-6, and score differences of at most 6e-8. For the untrained model, both checkpoints were exactly equal and gave the same replay AUROC, so the stored reference value comes from non-deterministic GPU execution in the frozen reference run, not from a code difference. These rows are accepted under the exception. Also recorded: the generalized copy runs with deterministic CUDA algorithms (`torch.use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`; commit `90205eb`), whereas the frozen HNSCC E1 run did not. This must be disclosed in the paper's Methods | None. It defines the numerical tolerance of the equivalence check only; no method, metric or hypothesis changes |
