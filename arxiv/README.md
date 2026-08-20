# SPECTRA-DA paper draft

This directory uses the official ICLR 2027 style files.

## Build

With Tectonic installed:

```bash
tectonic --keep-logs main.tex
```

Regenerate deterministic figures from the public aggregate metrics:

```bash
MPLCONFIGDIR="$PWD/.matplotlib" python scripts/make_figures.py
```

The current draft is intentionally conservative. It contains (i) an
equal-protocol Gate-1 real-target comparison on four tasks and 140 candidates
per task, where Transfer Score remains stronger in top-1 selection, and (ii) a
source-simulated refinement/ablation study on four tasks, 44 held-out shifts,
and 675 candidates per task. The pending externally operated 16-transfer
sealed evaluation is described as future evidence rather than a completed
result.

## GPT-image-2 architecture candidates

`notes/imagegen_prompts.jsonl` contains four prepared diagram briefs for the
bundled image-generation CLI. Generation requires `OPENAI_API_KEY` in the
environment, an OpenAI-compatible Images endpoint, and outbound network access.
The reviewed candidates are in `figures/gpt-image-2-candidates/`; candidate B
is the selected full-width Figure 1. See `notes/figure1_selection.md` for the
exact prompt and selection audit. The deterministic vector figure in
`figures/pipeline.pdf` remains available as a fallback.
