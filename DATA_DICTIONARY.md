# Data dictionary

All CSV files are UTF-8 with one header row. Labels use the eleven response
categories defined in the dissertation. Blank public-release fields indicate
removed transport metadata, not failed model output.

## Primary model outputs

Files: `gpt4o_all_results.csv`, `gpt5.6_all_results.csv`, and
`qwen36_flash_all_results.csv`.

- `image_id`, `filename`, `image_path`: corpus identifier and relative file
  reference. The image files are not distributed.
- `model`, `prompt_version`: requested model configuration and retained prompt
  version where available.
- `label`, `confidence`, `brief_reason`: parsed model response.
- `status`, `error`: request outcome.
- token, duration, and UTC timestamp fields: retained execution metadata.
- `raw_output`: retained provider text used for parsing and audit.

The public OpenAI files omit `response_id`. Qwen's source CSV did not contain a
response-ID column.

## Repeatability audit

`repeatability_subset_32.csv` records the fixed diagnostic subset and its four
selection strata. Each CSV under `repeatability_results/` contains 96 successful
rows: 32 images by three runs for one model.

The public copies preserve protocol hashes, model settings, labels, confidence,
reasons, run order, timestamps, and token counts. `subset_manifest` and
`image_path` are repository-relative. `response_id` and `raw_json_path` are
blank because provider IDs and raw envelopes are not distributed.

## Prompt-robustness audit

`prompt_engineering_experiment/results/prompt_experiment_results.csv` contains
432 successful classifications: three models by 16 images by three conditions
by three runs. `trial_id` is retained because it is the deterministic protocol
key used by the analyser. `execution_id`, `response_id`, and `raw_json_path` are
blank; `image_path` is repository-relative.

`prompt_engineering_experiment/data/human_endorsements_16.csv` contains 176
aggregate rows (16 images by 11 labels). It contains counts and rates only, with
no participant-level records or identifiers.

## Supporting manifests

- `data/researcher_working_labels.csv` contains only `image_id` and the working
  label used for the supplementary researcher-model comparison. Stale model
  columns from the source workbook are deliberately excluded.
- `data/image_manifest.csv` contains each of the 127 image IDs, original
  filename, SHA-256 checksum, and byte size. It contains no image pixels or
  source URLs.
- `canonical_127_image_ids.txt` fixes the full-corpus ID set.

## Restricted inputs

Participant-level survey data, the QSF, the image corpus, image variants, and
raw API request/response envelopes are held outside this public repository.
Consequently, the public files reproduce the reported aggregate analyses but
cannot rerun model inference or the participant-level bootstrap from scratch.
