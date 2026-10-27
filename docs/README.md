# Documentation

Every study was specified in a dated protocol before its generators were fitted. The protocol of each iteration states its hypotheses, data splits, arms and analyses; the runbook lists the commands that ran it. The SHA-256 digests of the protocols are recorded in `configs/` and in the run manifests, so the files keep their original names and locations.

## Studies reported in the paper

| Study | Protocol | Runbook | Results |
|---|---|---|---|
| Development case study (corrections and frozen comparison) | [ITERATION_11_PROTOCOL.md](ITERATION_11_PROTOCOL.md) | [ITERATION_11_RUNBOOK.md](ITERATION_11_RUNBOOK.md) | [reports/iteration11](../reports/iteration11) |
| Controlled study of training and evaluation pitfalls | [ITERATION_12_PROTOCOL.md](ITERATION_12_PROTOCOL.md) | [ITERATION_12_RUNBOOK.md](ITERATION_12_RUNBOOK.md) | [reports/iteration12](../reports/iteration12) |
| First relation test, NHANES 2017 to 2018 blood counts | [ITERATION_13_PROTOCOL.md](ITERATION_13_PROTOCOL.md) | [ITERATION_13_RUNBOOK.md](ITERATION_13_RUNBOOK.md) | [reports/iteration13](../reports/iteration13) |
| Robustness across survey cycles, relation types and sizes | [ITERATION_14_PROTOCOL.md](ITERATION_14_PROTOCOL.md) | [ITERATION_14_RUNBOOK.md](ITERATION_14_RUNBOOK.md) | [reports/iteration14](../reports/iteration14) |
| Decomposition of the training correction | [ITERATION_15_PROTOCOL.md](ITERATION_15_PROTOCOL.md) | | [reports/iteration15](../reports/iteration15) |
| Final comparison with knowledge-equivalent controls | | | [reports/final](../reports/final) |

## Development history

Iterations 2 to 10 developed the model on four UCI cohorts before the corrected protocol; their protocols and runbooks (`ITERATION_2_*` to `ITERATION_10_*`) and reports are kept for transparency.

## Other documents

| Document | Contents |
|---|---|
| [EXTERNAL_BASELINES.md](EXTERNAL_BASELINES.md) | How TabDDPM and TabDiff are run from their pinned upstream sources, and every adaptation |
| [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) | Implemented components and their tests |
| [MedHyperDiff_Research_Proposal_Revised.md](MedHyperDiff_Research_Proposal_Revised.md) | The research proposal; the original version is in `source/` |
| [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md), [PUBLICATION_READINESS.md](PUBLICATION_READINESS.md) | Early methodological review and evidence gates |
