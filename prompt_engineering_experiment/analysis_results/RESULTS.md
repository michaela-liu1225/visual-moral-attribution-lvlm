# Prompt engineering experiment results

Analysis status: **COMPLETE**.

Repeated calls were collapsed within each model, condition, and image. All intervals and tests use images as the inferential units; repeats are not counted as independent observations.

## Completeness

- Expected successful trials: 432
- Observed successful trials: 432
- Missing successful trials: 0
- Failure rows retained in trial logs: 0

## Condition summaries

| Model | Condition | Complete images | Expected endorsement (image macro) | Expected endorsement (response micro) | Modal-label support | Plurality matches | Stability | Mean confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt56 | P0 | 16/16 | 44.6% | 44.6% | 44.6% | 11/16 | 1.000 | 88.0 |
| gpt56 | P1 | 16/16 | 43.6% | 43.9% | 43.6% | 10/16 | 1.000 | 88.5 |
| gpt56 | P2 | 16/16 | 45.3% | 45.5% | 44.6% | 11/16 | 0.958 | 85.0 |
| gpt4o | P0 | 16/16 | 41.5% | 41.8% | 41.1% | 9/16 | 0.979 | 89.4 |
| gpt4o | P1 | 16/16 | 41.9% | 42.1% | 42.3% | 10/16 | 0.958 | 88.8 |
| gpt4o | P2 | 16/16 | 41.1% | 41.4% | 41.1% | 9/16 | 1.000 | 88.0 |
| qwen36 | P0 | 16/16 | 39.5% | 40.0% | 39.5% | 11/16 | 1.000 | 91.2 |
| qwen36 | P1 | 16/16 | 40.1% | 40.5% | 40.4% | 12/16 | 0.979 | 89.5 |
| qwen36 | P2 | 16/16 | 39.5% | 40.0% | 39.5% | 11/16 | 1.000 | 89.8 |

Expected human endorsement is the mean participant endorsement of the labels selected across repeats, calculated within each image and then macro-averaged across images for the primary estimand. The response-micro column is a pre-specified secondary summary weighted by each image's human n_responses. Stability is the modal label's share of successful repeats.

## Paired comparisons

| Model | Comparison | Paired images | Mean delta | 95% image-bootstrap CI | Exact sign-flip p | Improved / regressed | Plurality W→R / R→W | Exact McNemar p |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gpt56 | p2_minus_p0 (primary) | 16/16 | 0.7% | [-1.0%, 3.1%] | 1.0000 | 1 / 1 | 0 / 0 | 1.0000 |
| gpt56 | p1_minus_p0 | 16/16 | -1.0% | [-3.1%, 0.0%] | 1.0000 | 0 / 1 | 0 / 1 | 1.0000 |
| gpt56 | p2_minus_p1 | 16/16 | 1.7% | [0.0%, 4.5%] | 0.5000 | 2 / 0 | 1 / 0 | 1.0000 |
| gpt4o | p2_minus_p0 | 16/16 | -0.4% | [-1.2%, 0.0%] | 1.0000 | 0 / 1 | 0 / 0 | 1.0000 |
| gpt4o | p1_minus_p0 | 16/16 | 0.4% | [0.0%, 1.2%] | 1.0000 | 1 / 0 | 1 / 0 | 1.0000 |
| gpt4o | p2_minus_p1 | 16/16 | -0.8% | [-2.3%, 0.0%] | 1.0000 | 0 / 1 | 0 / 1 | 1.0000 |
| qwen36 | p2_minus_p0 | 16/16 | 0.0% | [0.0%, 0.0%] | 1.0000 | 0 / 0 | 0 / 0 | 1.0000 |
| qwen36 | p1_minus_p0 | 16/16 | 0.6% | [0.0%, 1.8%] | 1.0000 | 1 / 0 | 1 / 0 | 1.0000 |
| qwen36 | p2_minus_p1 | 16/16 | -0.6% | [-1.8%, 0.0%] | 1.0000 | 0 / 1 | 0 / 1 | 1.0000 |

The two-sided sign-flip test exactly enumerates all image-level sign assignments. The percentile confidence interval resamples images with replacement (20,000 draws; seed 20260905). McNemar's exact test uses only discordant modal-label plurality matches.

## Mixed-None sensitivity

The sensitivity analysis removes the three None endorsements that were co-selected with a moral category, as an alternative interpretation of those mixed responses. It uses the same image-level aggregation and paired tests.

### Condition summaries

| Model | Condition | Expected endorsement (image macro) | Expected endorsement (response micro) |
|---|---:|---:|---:|
| gpt56 | P0 | 44.6% | 44.6% |
| gpt56 | P1 | 43.6% | 43.9% |
| gpt56 | P2 | 45.3% | 45.5% |
| gpt4o | P0 | 41.5% | 41.8% |
| gpt4o | P1 | 41.9% | 42.1% |
| gpt4o | P2 | 41.1% | 41.4% |
| qwen36 | P0 | 39.5% | 40.0% |
| qwen36 | P1 | 40.1% | 40.5% |
| qwen36 | P2 | 39.5% | 40.0% |

### Paired comparisons

| Model | Comparison | Mean delta | 95% image-bootstrap CI | Exact sign-flip p |
|---|---|---:|---:|---:|
| gpt56 | p2_minus_p0 | 0.7% | [-1.0%, 3.1%] | 1.0000 |
| gpt56 | p1_minus_p0 | -1.0% | [-3.1%, 0.0%] | 1.0000 |
| gpt56 | p2_minus_p1 | 1.7% | [0.0%, 4.5%] | 0.5000 |
| gpt4o | p2_minus_p0 | -0.4% | [-1.2%, 0.0%] | 1.0000 |
| gpt4o | p1_minus_p0 | 0.4% | [0.0%, 1.2%] | 1.0000 |
| gpt4o | p2_minus_p1 | -0.8% | [-2.3%, 0.0%] | 1.0000 |
| qwen36 | p2_minus_p0 | 0.0% | [0.0%, 0.0%] | 1.0000 |
| qwen36 | p1_minus_p0 | 0.6% | [0.0%, 1.8%] | 1.0000 |
| qwen36 | p2_minus_p1 | -0.6% | [-1.8%, 0.0%] | 1.0000 |

## Interpretation limits

- This is an exploratory 16-image pilot, not an out-of-sample accuracy benchmark.
- Participant responses were multi-label endorsements; within-image rates need not sum to 100%.
- Confidence is model-reported and is not a calibrated probability.
- Repeated calls measure output stability but do not increase the inferential sample size above the number of images.
