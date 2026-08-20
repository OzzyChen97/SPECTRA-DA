# Figure 1 generation and selection

Generated on 2026-08-20 with `gpt-image-2`, high quality, through the
workspace AutoSOTA `config.yaml` research endpoint using
`tools/generate_gpt_image2_figure.py`. Credentials are not stored in the paper
repository.

## Candidate review

| Candidate | Scientific correctness | Text accuracy | Paper-size readability | Visual polish | Decision |
|---|---:|---:|---:|---:|---|
| A | 5/5 | 5/5 | 4/5 | 3/5 | Accurate and readable, but visually close to a plain block diagram. |
| B | 5/5 | 5/5 | 5/5 | 5/5 | Selected. It best separates label-free selection, source-only calibration, and sealed evaluation. |
| C | 4/5 | 4/5 | 4/5 | 4/5 | More visually rich, but adds equations/axis labels and extra details not needed in the overview. |
| D | 5/5 | 5/5 | 5/5 | 4/5 | Clean and accurate, but too compressed for the current method story. |

The paper uses candidate B as `figures/figure1-gpt-image-2.png`. The
deterministic `figures/pipeline.pdf` remains a local fallback/reference asset
only and is not referenced by the current draft.

## Exact selected prompt

> Use case: scientific-educational. Asset type: polished method overview figure for an ICLR paper. Primary request: visualize the SPECTRA-DA workflow as three horizontal regions. Region 1 header: Label-free target selection. Boxes: Candidate Bank; Target Predictions; Graph Spectral Frame; Band Agreements; Risk Recovery; Chosen Checkpoint. Region 2 header: Source calibration only. Boxes: Labeled Source Shifts; Unlabeled Matching; Error Covariance Transport. Region 3 header: Sealed evaluation. Boxes: Hidden Target Labels; Final Metrics. Show that source calibration feeds Risk Recovery, and hidden labels are only connected to Final Metrics after Chosen Checkpoint. Style: minimal academic vector diagram, high polish, strong whitespace, color-blind-safe blue-purple-orange-green-red palette, readable at paper width. Constraints: no arrows from Hidden Target Labels to any selector module, no SOTA wording, no result numbers, no equations, no watermark, no extra labels beyond the listed labels.
