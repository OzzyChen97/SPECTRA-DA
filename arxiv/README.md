# GDA-Select paper draft

This directory uses the official ICLR 2027 style files.

## Build

Use the repository-pinned Tectonic binary:

```bash
./tools/tectonic --keep-logs main.tex
```

Regenerate deterministic result/audit figures from the public aggregate
metrics:

```bash
MPLCONFIGDIR="$PWD/.matplotlib" python scripts/make_figures.py
```

The current draft is benchmark-first and intentionally conservative. It
contains (i) an equal-protocol real-target open-development comparison on four
tasks and 675 candidates per task, (ii) a source-simulated recovery/ablation
study over 44 held-out shifts, and (iii) the complete Stage A/B/C stopping
audit. No selector passed every promotion guard, so no method was frozen and
the final twelve transfers remain sealed.

## Architecture figure

The current full-width Figure 1 is the selected GPT-image-2 candidate stored as
`figures/figure1-gpt-image-2.png`. It was generated with
`tools/generate_gpt_image2_figure.py`, which reads the workspace AutoSOTA
`config.yaml` research endpoint and does not store credentials in this
repository. The four generated candidates are retained in
`figures/gpt-image-2-candidates-v2/`; see `notes/figure1_selection.md` for the
prompt and selection audit.
