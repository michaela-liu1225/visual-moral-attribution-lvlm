# Visual Moral Attribution in Large Vision-Language Models

This is the public reproducibility archive for Yuxin Liu's 2026 dissertation,
*Moral Alignment in Multimodal AI: Evaluating Large Vision-Language Models'
Capacity to Interpret the Moral Meaning of Images*.

The archive contains the model outputs, aggregate human-endorsement data,
analysis code, prompt conditions, frozen manifests, and a public access copy of
the dissertation with restricted images removed.

[Download the public access dissertation PDF](paper/dissertation_public_access.pdf).

## What is included

| Location | Contents |
| --- | --- |
| `paper/dissertation_public_access.pdf` | Dissertation text and tables with restricted images replaced by labelled placeholders and the student identifier removed. |
| `paper/source/` | LaTeX source for the public access copy, including placeholder figures. |
| `gpt4o_all_results.csv`, `gpt5.6_all_results.csv`, `qwen36_flash_all_results.csv` | The 127 primary classifications for each model. |
| `repeatability_results/` | Three-run outputs for the fixed 32-image subset and the reported summary. |
| `analysis/` | Chapter 4 recomputation, primary-to-repeat continuity, and the redacted image-edit console record. |
| `prompt_engineering_experiment/` | P0/P1/P2 prompt snapshots, manifests, 432 classifications, aggregate human targets, and final analysis outputs. |
| `data/` | Researcher working labels and an image ID/checksum manifest. No image pixels are included. |

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for field-level notes.

## Reproducing the public analyses

Python 3.10 or later is recommended. The repeatability and prompt-robustness
analyses use only the Python standard library.

```bash
python analyze_repeatability.py
python analysis/primary_repeat_continuity.py
python prompt_engineering_experiment/analyze_prompt_experiment.py \
  --models gpt56 gpt4o qwen36 \
  --output-dir prompt_engineering_experiment/analysis_results
```

The first and third commands regenerate the retained CSV and Markdown outputs.
The public release was checked by rerunning both analyses: the repeatability
summary and all five prompt-analysis CSV/Markdown outputs matched the retained
files byte for byte.

`analysis/chapter4_recompute.py` also covers the participant-level survey and
bootstrap calculations, but those parts require the restricted Qualtrics ZIP
passed with `--survey-zip`. That file is intentionally not public. The three
primary model CSVs and the redacted image-edit transcript remain usable without
the survey export.

## Public-release redactions

Only non-analytical transport metadata was removed from public CSV copies:
provider response IDs, execution IDs, absolute workstation paths, and raw-JSON
paths. Scientific fields such as image ID, model, prompt version, label,
confidence, brief reason, run number, timestamps, token counts, and protocol
hashes were retained. The image-edit transcript similarly omits terminal
user/host strings and provider response IDs while preserving the parsed outputs
used by the analysis.

## Data access, privacy, and third-party material

The image corpus, edited image variants, image-bearing survey export, raw QSF,
raw API envelopes, and participant-level Qualtrics export are not distributed.
The images include third-party or otherwise rights-restricted material, and the
participant-level export contains platform-generated identifiers and technical
metadata. Image IDs and checksums are retained for provenance; they do not grant
access or redistribution rights.

The public dissertation PDF is an edited access copy. Third-party images,
institutional artwork, and the student identifier have been removed. The
complete examination copy is not distributed in this repository.

## Licence scope

Copyright © 2026 Yuxin Liu. No blanket licence is granted for the repository.
In particular, no licence is granted for third-party names, trademarks,
photographs, institutional marks, or excluded material. Contact the author
before reusing code, data, or dissertation text beyond uses already permitted
by law.

## Citation

Please cite the dissertation and archive as described in
[`CITATION.cff`](CITATION.cff).
