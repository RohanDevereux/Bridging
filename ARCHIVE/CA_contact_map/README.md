# CA Contact Map

Archived contact-map / tensor-based branch.

Contents:

- `src/bridging/featurization`
  - MD-to-contact-map featurization using C-alpha interface patches and contact/distance channels
- `src/bridging/ml`
  - CVAE training over the contact-map tensors
- `src/bridging/regression`
  - Downstream regression and baseline-comparison code on top of the learned latent space
- `hpc/`
  - HPC launchers for featurization and model training
- `tests/`
  - Old contact-map smoke tests and artifacts

This branch predates the final graph-based pipeline and is no longer part of the active report workflow.
