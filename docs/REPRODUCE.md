# Reproduction Guide

This document specifies the exact environment, execution commands, and configuration details required to reproduce all stages of the multi-omics graph autoencoder benchmark across TCGA-HNSC and TCGA-BRCA cohorts.

---

## Provenance of Published Runs

The published benchmark results were generated from the following source code commits:

| Cancer Cohort | Pipeline Engine | Original Commit | Pinned Dataset Revision | Expected Output Directory |
|---|---|---|---|---|
| **TCGA-HNSC** ($n=515$) | `graphr1` (frozen E1) | `df42798` | `PUBLIC_DATASET_REVISION` | `outputs/v2/hnscc_v2_e1/` |
| **TCGA-BRCA** ($n=1{,}082$) | `graphr1g` (generalized E5) | `552c85d` | `PUBLIC_DATASET_REVISION` | `outputs/v2/brca_v2_e5/` |

---

## 1. Environment & Test Suite

Verify that all dependencies and primitive checks pass:

```bash
pip install -r requirements-v2.txt
pytest tests/
```

---

## 2. Smoke Testing (CPU)

To test the complete execution loop (cube building, cross-validation splits, Hallmark node ranking, graph generation, model training, baselines, and evaluation) within ~3 minutes:

```bash
python3 scripts/run_v2g.py --config configs/public/brca_smoke.yaml --stages cube,folds,nodes,graphs,runs,baselines,eval
```

Expected output directory: `outputs/public/brca_v2_smoke/`

---

## 3. Full Benchmark Execution

### TCGA-BRCA Full Benchmark

To run the complete BRCA benchmark across all 15 splits on an NVIDIA T4 GPU instance:

```bash
bash scripts/run_v2g_gcloud.sh --config configs/public/brca_e5.yaml
```

Alternatively, to execute locally with strict resource bounding (capped at 5 GB RAM and 6 CPU cores):

```bash
bash scripts/run_v2g_bounded.sh --config configs/public/brca_e5.yaml
```

Expected output directory: `outputs/v2/brca_v2_e5/`

---

### TCGA-HNSC Full Benchmark

The original HNSCC E1 benchmark was executed using the frozen `graphr1` package via `scripts/run_v2.py`. Because `graphr1/config.py` was frozen without declarative Hugging Face schema keys in `configs/v2/base.yaml`, the input dataset is fetched via environment variables or loaded through the generalized `graphr1g` engine.

#### Option A: Running with Generalized Pipeline (`graphr1g`)
`graphr1g` contains native support for the declarative `data.hf` configuration and has been verified to be mathematically equivalent:

```bash
bash scripts/run_v2g_gcloud.sh --config configs/public/hnscc_v2g.yaml
```

Expected output directory: `outputs/v2/hnscc_v2_e1/`

#### Option B: Running with Frozen Pipeline (`graphr1`)
To run using the original frozen `scripts/run_v2_gcloud.sh` script, export the target dataset repository before execution:

```bash
export HF_DATASET_REPO="HawkFranklin-Research/omics-graph-priors-tcga"
export HF_DATASET_REVISION="PUBLIC_DATASET_REVISION"
bash scripts/run_v2_gcloud.sh --config configs/v2/hnscc_gcloud.yaml
```

Or for direct local execution after downloading data to `data/hf_tcga`:

```bash
python3 scripts/run_v2.py \
    --config configs/v2/hnscc_gcloud.yaml \
    data.processed_root=data/hf_tcga \
    data.reference_root=data/hf_tcga/references \
    --stages all
```

Expected output directory: `outputs/v2/hnscc_v2_e1/`

---

## 4. Follow-Up Analysis Tables

Once both `outputs/v2/hnscc_v2_e1/` and `outputs/v2/brca_v2_e5/` are populated, generate the consolidated diagnostic and follow-up tables:

```bash
python3 scripts/analysis/followup_tables.py
python3 scripts/analysis/followup_degree_recon.py
python3 scripts/analysis/followup_attention_degree.py
python3 scripts/analysis/followup_esr1_diagnostic.py
```

---

## 5. Execution Characteristics & Computational Notes

1. **State Persistence and Resumption:**
   - Completed units, fold splits, and trained model checkpoints are checked on disk before execution. If a run is interrupted or preempted, re-executing the runner command skips already completed units and resumes where execution halted.
2. **Deterministic Numerics Policy:**
   - Both pipelines enforce strict PyTorch deterministic mode (`torch.use_deterministic_algorithms(True)`) and deterministic cuBLAS reduction workspaces (`CUBLAS_WORKSPACE_CONFIG=:4096:8`). This ensures bit-level reproducibility across identical GPU architectures, but reduces raw GPU throughput compared to non-deterministic modes.
3. **Attribution Stage Re-computation:**
   - While model training (`05_runs`) resumes per individual unit upon interruption, the integrated gradients attribution stage (`10_attribution`) operates per target block and evaluates all units for an incomplete target upon restart.
