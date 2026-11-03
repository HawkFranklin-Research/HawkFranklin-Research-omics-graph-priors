# HNSCC v2 pre-specified protocol

The reference condition is `gatv2_string`. The primary metrics are NM-R²(rna), HPV AUROC (readout `gene_latent_pca`) and OS ΔC.

| Hypothesis | Comparison | Question |
|---|---|---|
| H1 graph vs no graph | `gatv2_string` vs `dense_ae`, `node_mlp`, `pca_concat`, `lowrank_impute` | Does a gene graph beat graph-free models that see the same data? |
| H2 topology | `gatv2_string` vs `gatv2_string_rewired` | Is the *specific* biological wiring what matters? |
| H3 attention | `gatv2_string` vs `gcn_string` (and `sage_string`) | Does dynamic attention help beyond fixed aggregation? |
| H4 prior vs data graph | `gatv2_string` vs `gatv2_coexpr` | Is prior knowledge better than (density-matched) co-expression? |

Holm correction is applied within each primary metric across H1–H4. Everything else is labelled exploratory. The primary patient readout is `gene_latent_pca`; mean pooling is secondary. Node variance selection across all patients is label-blind and must be disclosed. CPTAC is excluded from the primary analysis because modality and endpoint coverage are insufficient.
