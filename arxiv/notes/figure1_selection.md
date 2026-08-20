# Figure 1 generation and selection

Generated on 2026-08-20 with `gpt-image-2`, high quality, through the AutoSOTA-configured OpenAI-compatible research endpoint. Credentials are not stored in the paper repository.

## Candidate review

| Candidate | Scientific correctness | Text accuracy | Paper-size readability | Visual polish | Decision |
|---|---:|---:|---:|---:|---|
| A | 5/5 | 5/5 | 3/5 | 3/5 | Accurate but visually sparse and text-heavy. |
| B | 5/5 | 5/5 | 4/5 | 5/5 | **Selected.** Clearest separation of target-label-free selection, source-only calibration, and sealed evaluation. |
| C | 4/5 | 5/5 | 4/5 | 4/5 | Attractive teaser, but gradients and decorative density are less suitable for a camera-ready method figure. |
| D | 4/5 | 5/5 | 5/5 | 5/5 | Strong typography, but its sequential stage arrows make the calibration dependency less precise than B. |

The paper uses `figures/gpt-image-2-candidates/pipeline-gpt2-b.png` as a full-width Figure 1. The previous deterministic figure is retained as a fallback.

## Exact selected prompt

> Use case: scientific-educational. Asset type: two-lane ICLR paper method overview. Primary request: visualize SPECTRA-DA with an upper Target-Label-Free Selection lane and a lower Source Calibration lane. Upper lane labels, verbatim: Candidate Trajectories; Target Graph and Predictions; Tight Spectral Bands; Pair Disagreement; Covariance-Corrected Risk; Select Candidate. Lower lane labels, verbatim: Source Graph plus Labels; Simulated Graph Shifts; Shift Descriptors; Band Covariance Bank; Uncertainty. Connect Shift Descriptors and Band Covariance Bank to Covariance-Corrected Risk. Place a locked box outside both lanes labeled Sealed Target Labels; One-Time Evaluator, with no arrow into the selector. Style: camera-ready scientific vector diagram, precise alignment, minimal ink, color-blind-safe palette, white background, strong hierarchy, readable labels. Constraints: no people, no devices, no photorealism, no 3D, no equations, no extra text, no watermark, preserve the information boundary exactly.

