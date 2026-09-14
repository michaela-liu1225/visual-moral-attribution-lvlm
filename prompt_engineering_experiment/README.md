# Exploratory P0/P1/P2 prompt-robustness audit

This directory contains the completed exploratory prompt audit reported in the
dissertation. The frozen design produced 432 successful classifications:

```text
3 models × 16 images × 3 prompt conditions × 3 repeats = 432
```

The unit of inference is the image (`n = 16`); the three repeats are averaged
within each image-condition cell.

| Condition | Role | Change from the common prompt |
| --- | --- | --- |
| P0 | Retained baseline | No additional decision procedure. |
| P1 | Structured control | Adds a generic step-by-step decision procedure without cue-specific weighting. |
| P2 | Visual-cue-guided condition | Adds the frozen evidence-weighting rules derived from non-overlapping image-edit cases. |

P1 and P2 are parallel variants of P0, not cumulative prompts. Their exact
texts are under `prompts/`, and the frozen schedule and evaluation set are
under `manifests/`.

## Reproduce the analysis

Run from the repository root:

```bash
python prompt_engineering_experiment/analyze_prompt_experiment.py \
  --models gpt56 gpt4o qwen36 \
  --output-dir prompt_engineering_experiment/analysis_results
```

The command reads the public 432-row result CSV and the anonymous 16-by-11
human-endorsement table. It regenerates the files under `analysis_results/`.

The image corpus, live inference runner, raw API envelopes, and provider IDs
are not included in this public release. The frozen runner's SHA-256 remains in
`protocol_manifest.json` as a provenance record. The public archive therefore
supports analysis reproduction, not a fresh inference run.

See `PROTOCOL.md` for the estimand, repeat aggregation, bootstrap, exact
sign-flip test, tie handling, and interpretation limits.
