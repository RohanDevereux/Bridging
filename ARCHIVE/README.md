# ARCHIVE

This directory contains work that is no longer part of the active final-report pipeline.

Active flow:

`PPB-Affinity dataset -> OpenMM MD -> GraphVAE dataset preparation -> GraphVAE training / sweeps / resampling`

Everything here is preserved for appendix material.

## Archived Workstreams

- [`CA_contact_map/`](CA_contact_map/README.md)
  - First-frame / contact-map featurization
  - CVAE and regression pipeline built on those tensors
- [`MMGBSA/`](MMGBSA/README.md)
  - AmberTools MM/GBSA prefetch, merge, and correction baseline tooling
- [`PRODIGY/`](PRODIGY/README.md)
  - PRODIGY baseline tooling and cache/prefetch path
- [`PPB_Affinity_augmentation/`](PPB_Affinity_augmentation/README.md)
  - PPB-style frame, trajectory, and ensemble models
  - Vendored `BaselineModel` code used by that branch
- [`Data_curation/`](Data_curation/README.md)
  - Old dataset filtering, size-prefetch, and subset-selection utilities

The active code remains under `src/bridging/MD`, `src/bridging/graphvae`, `src/bridging/utils`, and the small active subset of `src/bridging/dataProcessing`.
