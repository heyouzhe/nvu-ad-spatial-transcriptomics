# NVU AD Spatial Transcriptomics

Neurovascular dysfunction, glial activation and synaptic failure are central
features of Alzheimer's disease (AD), but how these processes are organized
within local human tissue microenvironments remains unresolved. In this study,
vascular-centered, communication-qualified digital neurovascular units (NVUs)
were reconstructed from single-cell-resolution Stereo-seq profiles of the human
hippocampus and prefrontal cortex, spanning 44 tissue sections from 28 donors.
Digital NVUs were anchored to vascular cores and refined by local
neuronal-astroglial context and ligand-receptor connectivity.

AD-associated remodeling of digital NVUs was anatomically patterned rather than
uniform. Hippocampal FAS, SLRM and CA1 showed increased digital NVU density,
region-specific compositional shifts and vascular-immune-stress activation
programs, whereas vulnerable prefrontal cortical NVUs in L456 and L23 were
dominated by suppression of neuronal synaptic, translational and
mitochondrial-energy-related programs. Aβ-proximal hippocampal NVU compartments
marked local remodeling hotspots, and a disease-informed hierarchical graph
neural network integrated cell-level, NVU-level and region-level features to
resolve region-by-disease tissue states while preserving hippocampal-cortical
structure.

![Figure 1 workflow schematic](docs/figures/figure1.png)

## Figure Code

| Manuscript figure | Main analysis | Code |
| --- | --- | --- |
| Figure 1 | Vascular-centered digital NVU reconstruction and vascular-field visualization | `notebooks/figure1_nvu_reconstruction.ipynb`; `scripts/figure1_vascular_ficture.py`; `docs/figures/figure1.png` |
| Figure 2 | AD-associated digital NVU abundance and cellular composition changes | `notebooks/figure2_ad_nvu_abundance_composition.ipynb` |
| Figure 3 | Hippocampal and cortical DEG, hdWGCNA, enrichment, and hub-gene network analyses | `notebooks/figure3_hippocampus_wgcna_up.ipynb`; `notebooks/figure3_hippocampus_wgcna_down.ipynb`; `notebooks/figure3_cortex_wgcna_up.ipynb`; `notebooks/figure3_cortex_wgcna_down.ipynb` |
| Figure 4 | Stereosite ligand-receptor communication landscapes | `notebooks/figure4_hippocampus_stereosite_allpairs.ipynb`; `notebooks/figure4_cortex_stereosite_allpairs.ipynb` |
| Figure 5 | Disease-associated astrocyte and microglial state analyses | `notebooks/figure5_disease_associated_glia.ipynb` |
| Figure 6 | Aβ-associated NVU remodeling, density, and gene-change analyses | `notebooks/figure6_abeta_nvu_gene_changes.ipynb`; `notebooks/figure6_abeta_nvu_integrated_changes.ipynb` |
| Figure 7 | Multi-scale GNN vulnerability modeling and interpretation | `scripts/figure7_model.py` for the training workflow; `notebooks/figure7_gnn_vulnerability_modeling.ipynb` for figure-panel plotting |
