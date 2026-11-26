# Provenance

This package was built from pre-existing, aggregate result outputs in the
manuscript workspace. It was curated on 2026-10-21 for public/repository
sharing.

- Seeds: 42--51.
- Internal endpoint: TCGA progression-free interval (PFI).
- External endpoint: METABRIC recurrence-free survival (RFS).
- Feature analyses: fixed-contrast WFP summaries, including the matching
  fixed-support treatment-profile/treatment-pair summaries; see
  `METHOD_AND_VERSION_NOTES.md`.
- No raw TCGA/METABRIC data, individual identifiers, individual outcomes,
  individual predictions, molecular input matrices, or checkpoints were copied.

`scripts/build_results_release.py` lists every approved local source by path
relative to the manuscript workspace. It intentionally fails if a named source
is missing, rather than searching broadly and accidentally adding a
patient-level artifact. `CHECKSUMS.sha256` hashes the released files.

An archived August 2026 recovery rerun was reviewed separately. It contains
partial DGP split summaries and an alternate ten-seed baseline archive, but is
not part of this release because it has not been reconciled to the submitted
internal-performance tables. See `NOT_FOUND_OR_NOT_RELEASED.md`.
