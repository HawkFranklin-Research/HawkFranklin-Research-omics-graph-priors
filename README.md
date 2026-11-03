# omics-graph-priors

Reproduction code for a controlled two-cancer benchmark evaluating biological gene networks in multiomics graph autoencoders against data-derived graphs and low-rank baselines.

---

## Benchmark Design

This benchmark provides a leak-free evaluation of prior biological gene networks across two epithelial cancers from The Cancer Genome Atlas (TCGA): Head and Neck Squamous Cell Carcinoma (HNSCC, $n = 515$) and Breast Invasive Carcinoma (BRCA, $n = 1{,}082$). For each cohort, a harmonized 4-channel tensor ($2{,}000$ MSigDB Hallmark genes $\times 4$ modalities: mRNA expression, promoter DNA methylation, copy-number alteration, and somatic mutation) is processed across 3 repeats of 5-fold cross-validation (15 splits per cohort). Evaluated conditions comprise message-passing graph autoencoders (GATv2, GCN, GraphSAGE) using external biological networks (STRING v12, combined score $\ge 700$), empirical training-fold co-expression graphs, matched degree-preserved rewired graph nulls, untrained initialization nulls, and graph-free baselines (per-gene MLPs, dense patient autoencoders, and low-rank linear baselines).

---

## Repository Layout

```
.
├── graphr1/             # Frozen HNSCC E1 pipeline package (commit df42798)
├── graphr1g/            # Generalized multi-cancer pipeline package (BRCA E5)
├── configs/
│   ├── v2/              # Frozen HNSCC configs and pre-registration protocols
│   ├── v2g/             # Generalized configurations (BRCA and equivalence runs)
│   └── public/          # Overlay configs pointing to public Hugging Face assets
├── scripts/
│   ├── run_v2.py        # HNSCC frozen pipeline CLI runner
│   ├── run_v2_bounded.sh# Resource-bounded runner (5 GB RAM, 6 CPU cores)
│   ├── run_v2_gcloud.sh # GCP Spot execution runner for HNSCC
│   ├── run_v2g.py       # Generalized multi-cancer CLI runner
│   ├── run_v2g_bounded.sh
│   ├── run_v2g_gcloud.sh# GCP Spot execution runner for BRCA
│   ├── compare_v2_runs.py
│   └── analysis/        # Follow-up analysis scripts (degree, attention, tables)
├── tests/               # Test suite for configuration, protocol, and primitives
├── docs/
│   └── REPRODUCE.md     # Step-by-step reproduction guide and execution notes
├── requirements-v2.txt  # Pinned Python package dependencies
├── LICENSE              # Apache-2.0
├── CITATION.cff         # Citation file format
└── .zenodo.json         # Zenodo archive metadata
```

---

## Installation

Requirements: Python 3.10+ with PyTorch and PyTorch Geometric.

```bash
git clone https://github.com/HawkFranklin-Research/omics-graph-priors.git
cd omics-graph-priors
pip install -r requirements-v2.txt
```

---

## Data & Checkpoints

- **Benchmark Dataset:** [HawkFranklin-Research/omics-graph-priors-tcga](https://huggingface.co/datasets/HawkFranklin-Research/omics-graph-priors-tcga)
  - Pinned Revision: `631cac47ae2cd16fae171dcaef2ec7f77c6b9004`
- **Trained Model Checkpoints:** [HawkFranklin-Research/omics-graph-priors-models](https://huggingface.co/HawkFranklin-Research/omics-graph-priors-models)

---

## Quickstart: Local Smoke Run (CPU)

To verify the installation and end-to-end execution on a reduced dataset using CPU:

```bash
python3 scripts/run_v2g.py --config configs/public/brca_smoke.yaml --stages cube,folds,nodes,graphs,runs,baselines,eval
```

---

## Full Benchmark Execution (GCP Spot T4 GPU)

The full cross-validation runs execute across 15 splits per cohort on an NVIDIA T4 GPU instance (`n1-standard-8`, 16 GB GPU, 30 GB RAM). Approximate VM compute durations:
- **TCGA-HNSC:** ~8.2 VM hours (`df42798`)
- **TCGA-BRCA:** ~29.9 VM hours (`552c85d`)

### Execution Commands

For BRCA:
```bash
bash scripts/run_v2g_gcloud.sh --config configs/public/brca_e5.yaml
```

For HNSCC:
```bash
bash scripts/run_v2g_gcloud.sh --config configs/public/hnscc_v2g.yaml
```

See [docs/REPRODUCE.md](docs/REPRODUCE.md) for detailed execution options, environment overrides, and frozen runner instructions.

---

## Pipeline Output Stages

Pipeline execution generates standardized stage outputs within `outputs/v2/<experiment_name>/`:

| Stage | Name | Description |
|---|---|---|
| `01_cube` | Data Harmonization | 4-channel tensor assembly (`cube.npz`), patient and gene tables |
| `02_folds` | Cross-Validation | Leak-free stratified 5-fold splits repeated across 3 random seeds |
| `03_nodes` | Gene Universe | MSigDB Hallmark variance ranking and top 2,000 gene node selection |
| `04_graphs` | Graph Topologies | STRING v12 extraction, training-fold co-expression, and rewired nulls |
| `05_runs` | Model Training | 10 model arms trained across 15 splits (checkpoints, metrics, latents) |
| `06_baselines` | Low-Rank Baselines | Linear SVD, mean baselines, and feature extractors |
| `07_eval` | Performance Metrics | Masked reconstruction R², clinical subtype AUROC, survival C-index |
| `08_circularity` | Leakage Audit | Cross-fold stability checks and data overlap diagnostics |
| `09_attention` | Attention Analysis | GATv2 attention matrix extraction, stability, and topology tests |
| `10_attribution`| Integrated Gradients | In silico knockout attribution and gradient importance scores |
| `11_enrichment` | Pathway Analysis | GSEA and MSigDB hypergeometric functional enrichment |
| `12_report` | Summary Reports | Consolidated performance markdown tables and protocol compliance |

---

## Pre-Registration Protocols

Experimental procedures and primary evaluation gates were registered prior to model execution:
- **HNSCC:** [`configs/v2/protocol_hnscc.md`](configs/v2/protocol_hnscc.md)
  - SHA-256: `671c273c450a1cdffa6bb4250be597210ebf36ffa2d2ae66e83ed9340f194588`
- **BRCA:** [`configs/v2/protocol_brca.md`](configs/v2/protocol_brca.md)
  - SHA-256: `374341db16a4b4f077f216786f8eb56f5cd3dd87f463ad2780ec47f0502978ed`

---

## Citation

```bibtex
@article{Patel2026OmicsGraphPriors,
  title   = {Controlled benchmark of biological gene graphs in multiomics graph autoencoders on TCGA cancers},
  author  = {Patel, Vatsal},
  year    = {2026},
  note    = {Under review}
}
```

---

## License

This software and replication suite is licensed under the [Apache License 2.0](LICENSE).
