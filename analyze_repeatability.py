from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from repeatability_subset_common import DEFAULT_EXPERIMENT_ID, VALID_LABELS


NONE_LABEL = "None of these / Not clearly moral in nature"
FOUNDATION_MAP = {
    "Care": "Care/Harm",
    "Harm": "Care/Harm",
    "Fairness": "Fairness/Cheating",
    "Cheating": "Fairness/Cheating",
    "Loyalty": "Loyalty/Betrayal",
    "Betrayal": "Loyalty/Betrayal",
    "Respect / Authority": "Authority/Subversion",
    "Subversion": "Authority/Subversion",
    "Sanctity": "Sanctity/Degradation",
    "Degradation": "Sanctity/Degradation",
    NONE_LABEL: "None",
}


def exact(label: str) -> str:
    return label


def foundation(label: str) -> str:
    return FOUNDATION_MAP[label]


def moral_binary(label: str) -> str:
    return "None" if label == NONE_LABEL else "Moral label"


RESOLUTIONS: dict[str, Callable[[str], str]] = {
    "exact_label": exact,
    "foundation": foundation,
    "moral_vs_none": moral_binary,
}


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    categories = set(counts_a) | set(counts_b)
    expected = sum(
        (counts_a[category] / n) * (counts_b[category] / n)
        for category in categories
    )
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def fleiss_kappa(ratings: list[list[str]]) -> float | None:
    if not ratings:
        return None
    raters = len(ratings[0])
    if raters < 2 or any(len(row) != raters for row in ratings):
        return None

    categories = sorted({label for row in ratings for label in row})
    item_agreement = []
    overall_counts = Counter()
    for row in ratings:
        counts = Counter(row)
        overall_counts.update(row)
        item_agreement.append(
            (sum(count * count for count in counts.values()) - raters)
            / (raters * (raters - 1))
        )

    observed = statistics.mean(item_agreement)
    total = len(ratings) * raters
    expected = sum((overall_counts[c] / total) ** 2 for c in categories)
    if expected == 1:
        return 1.0 if observed == 1 else None
    return (observed - expected) / (1 - expected)


def agreement(labels_a: list[str], labels_b: list[str]) -> float | None:
    if not labels_a or len(labels_a) != len(labels_b):
        return None
    return sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def load_results(
    path: Path,
    experiment_id: str,
) -> tuple[str, dict[tuple[int, str], str], dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Result CSV not found: {path.resolve()}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    rows = [
        row
        for row in rows
        if row.get("experiment_id") == experiment_id
        and row.get("status") == "success"
    ]
    if not rows:
        raise ValueError(
            f"No successful rows for experiment {experiment_id!r} in {path}."
        )

    requested_models = {row["requested_model"] for row in rows}
    if len(requested_models) != 1:
        raise ValueError(f"Mixed requested models in {path}: {requested_models}")
    model = next(iter(requested_models))

    labels: dict[tuple[int, str], str] = {}
    duplicate_keys: list[tuple[int, str]] = []
    for row in rows:
        key = (int(row["run_number"]), row["image_id"])
        label = row["label"]
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label in {path}: {label!r}")
        if key in labels:
            duplicate_keys.append(key)
        labels[key] = label

    metadata = {
        "path": str(path.resolve()),
        "successful_rows": len(rows),
        "unique_results": len(labels),
        "duplicate_success_keys_replaced_by_last_row": duplicate_keys,
        "actual_models": sorted(
            {row.get("actual_model", "") for row in rows if row.get("actual_model")}
        ),
        "manifest_hashes": sorted(
            {
                row.get("subset_manifest_sha256", "")
                for row in rows
                if row.get("subset_manifest_sha256")
            }
        ),
        "prompt_versions": sorted(
            {row.get("prompt_version", "") for row in rows}
        ),
        "prompt_hashes": sorted(
            {row.get("prompt_sha256", "") for row in rows}
        ),
        "temperature_values": sorted(
            {row.get("temperature_requested", "") for row in rows}
        ),
        "reasoning_modes": sorted({row.get("reasoning_mode", "") for row in rows}),
        "time_range_utc": [
            min(row["request_started_at_utc"] for row in rows),
            max(row["response_received_at_utc"] for row in rows),
        ],
    }
    return model, labels, metadata


def within_model_summary(
    labels: dict[tuple[int, str], str],
) -> dict[str, object]:
    runs = sorted({run for run, _ in labels})
    image_sets = [
        {image_id for run, image_id in labels if run == run_number}
        for run_number in runs
    ]
    complete_images = sorted(set.intersection(*image_sets)) if image_sets else []

    output: dict[str, object] = {
        "runs": runs,
        "complete_images": len(complete_images),
        "resolutions": {},
    }
    for resolution, transform in RESOLUTIONS.items():
        pair_results = []
        for run_a, run_b in itertools.combinations(runs, 2):
            first = [transform(labels[(run_a, image)]) for image in complete_images]
            second = [transform(labels[(run_b, image)]) for image in complete_images]
            pair_results.append(
                {
                    "runs": [run_a, run_b],
                    "n": len(complete_images),
                    "agreement": rounded(agreement(first, second)),
                    "cohen_kappa": rounded(cohen_kappa(first, second)),
                }
            )

        ratings = [
            [transform(labels[(run, image)]) for run in runs]
            for image in complete_images
        ]
        unanimous_count = sum(len(set(row)) == 1 for row in ratings)
        pairwise_rates = [
            row["agreement"]
            for row in pair_results
            if row["agreement"] is not None
        ]
        output["resolutions"][resolution] = {
            "pairwise_runs": pair_results,
            "mean_pairwise_agreement": (
                rounded(statistics.mean(pairwise_rates)) if pairwise_rates else None
            ),
            "min_pairwise_agreement": min(pairwise_rates) if pairwise_rates else None,
            "max_pairwise_agreement": max(pairwise_rates) if pairwise_rates else None,
            "unanimous_images": unanimous_count,
            "unanimous_rate": rounded(
                unanimous_count / len(complete_images) if complete_images else None
            ),
            "fleiss_kappa": rounded(fleiss_kappa(ratings)),
        }
    return output


def unique_modes(
    labels: dict[tuple[int, str], str],
) -> tuple[dict[str, str], list[str]]:
    by_image: dict[str, list[str]] = defaultdict(list)
    for (_, image_id), label in labels.items():
        by_image[image_id].append(label)

    modes: dict[str, str] = {}
    unresolved: list[str] = []
    for image_id, values in by_image.items():
        counts = Counter(values)
        highest = max(counts.values())
        candidates = [label for label, count in counts.items() if count == highest]
        if len(candidates) == 1:
            modes[image_id] = candidates[0]
        else:
            unresolved.append(image_id)
    return modes, sorted(unresolved)


def inter_model_summary(
    first: dict[tuple[int, str], str],
    second: dict[tuple[int, str], str],
) -> dict[str, object]:
    common_keys = sorted(set(first) & set(second))
    first_modes, first_unresolved = unique_modes(first)
    second_modes, second_unresolved = unique_modes(second)
    common_mode_images = sorted(set(first_modes) & set(second_modes))

    output: dict[str, object] = {
        "matched_run_image_comparisons": len(common_keys),
        "modal_comparable_images": len(common_mode_images),
        "first_unresolved_modal_images": first_unresolved,
        "second_unresolved_modal_images": second_unresolved,
        "resolutions": {},
    }

    for resolution, transform in RESOLUTIONS.items():
        matched_first = [transform(first[key]) for key in common_keys]
        matched_second = [transform(second[key]) for key in common_keys]
        modal_first = [transform(first_modes[image]) for image in common_mode_images]
        modal_second = [transform(second_modes[image]) for image in common_mode_images]
        output["resolutions"][resolution] = {
            "matched_runs": {
                "n": len(common_keys),
                "agreement": rounded(agreement(matched_first, matched_second)),
                "cohen_kappa": rounded(
                    cohen_kappa(matched_first, matched_second)
                ),
            },
            "unique_modal_labels": {
                "n": len(common_mode_images),
                "agreement": rounded(agreement(modal_first, modal_second)),
                "cohen_kappa": rounded(cohen_kappa(modal_first, modal_second)),
            },
        }
    return output


def build_tidy_rows(summary: dict[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, data in summary["within_model"].items():
        for resolution, values in data["resolutions"].items():
            rows.append(
                {
                    "scope": "within_model",
                    "comparison": model,
                    "statistic": "mean_pairwise_run_agreement",
                    "resolution": resolution,
                    "n": data["complete_images"],
                    "agreement_rate": values["mean_pairwise_agreement"],
                    "kappa": values["fleiss_kappa"],
                    "notes": (
                        f"unanimous={values['unanimous_images']}/"
                        f"{data['complete_images']}"
                    ),
                }
            )
    for comparison, data in summary["inter_model"].items():
        for resolution, values in data["resolutions"].items():
            for statistic, metric in values.items():
                rows.append(
                    {
                        "scope": "inter_model",
                        "comparison": comparison,
                        "statistic": statistic,
                        "resolution": resolution,
                        "n": metric["n"],
                        "agreement_rate": metric["agreement"],
                        "kappa": metric["cohen_kappa"],
                        "notes": "",
                    }
                )
    return rows


def write_tidy_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
        "comparison",
        "statistic",
        "resolution",
        "n",
        "agreement_rate",
        "kappa",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Report within-model repeatability and compare it with "
            "inter-model agreement."
        )
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument(
        "--gpt4o-csv",
        type=Path,
        default=Path("repeatability_results/gpt4o_subset_repeats.csv"),
    )
    parser.add_argument(
        "--gpt56-csv",
        type=Path,
        default=Path("repeatability_results/gpt56_subset_repeats.csv"),
    )
    parser.add_argument(
        "--qwen36-csv",
        type=Path,
        default=Path("repeatability_results/qwen36_subset_repeats.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("repeatability_results/repeatability_summary.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("repeatability_results/repeatability_summary.csv"),
    )
    args = parser.parse_args()

    inputs = [args.gpt4o_csv, args.gpt56_csv, args.qwen36_csv]
    model_results: dict[str, dict[tuple[int, str], str]] = {}
    input_metadata: dict[str, object] = {}
    for path in inputs:
        model, labels, metadata = load_results(path, args.experiment_id)
        if model in model_results:
            raise ValueError(f"Duplicate model input: {model}")
        model_results[model] = labels
        input_metadata[model] = metadata

    manifest_hashes = {
        value
        for metadata in input_metadata.values()
        for value in metadata["manifest_hashes"]
    }
    if len(manifest_hashes) != 1:
        raise ValueError(
            "The three result files do not use one identical subset manifest: "
            f"{sorted(manifest_hashes)}"
        )

    prompt_versions = {
        value
        for metadata in input_metadata.values()
        for value in metadata["prompt_versions"]
    }
    if len(prompt_versions) != 1:
        raise ValueError(
            "The three result files do not use one prompt version: "
            f"{sorted(prompt_versions)}"
        )

    prompt_hashes = {
        value
        for metadata in input_metadata.values()
        for value in metadata["prompt_hashes"]
    }
    if len(prompt_hashes) != 1:
        raise ValueError(
            "The three result files do not use one exact prompt text: "
            f"{sorted(prompt_hashes)}"
        )

    temperature_values = {
        value
        for metadata in input_metadata.values()
        for value in metadata["temperature_values"]
    }
    if len(temperature_values) != 1:
        raise ValueError(
            "The three result files do not use one requested temperature: "
            f"{sorted(temperature_values)}"
        )

    key_sets = {model: set(labels) for model, labels in model_results.items()}
    first_key_set = next(iter(key_sets.values()))
    if any(keys != first_key_set for keys in key_sets.values()):
        raise ValueError(
            "The three models do not have the same successful "
            "(run_number, image_id) cells. Complete or retry missing cells "
            "before the formal comparison."
        )

    runs = sorted({run_number for run_number, _ in first_key_set})
    if not 3 <= len(runs) <= 5:
        raise ValueError(
            f"Formal analysis requires 3-5 completed runs; found {runs}."
        )
    run_sizes = Counter(run_number for run_number, _ in first_key_set)
    if set(run_sizes.values()) != {32}:
        raise ValueError(
            "Every run must contain all 32 fixed subset images; found "
            f"{dict(sorted(run_sizes.items()))}."
        )

    within = {
        model: within_model_summary(labels)
        for model, labels in model_results.items()
    }
    inter = {}
    for first_model, second_model in itertools.combinations(model_results, 2):
        key = f"{first_model} | {second_model}"
        inter[key] = inter_model_summary(
            model_results[first_model],
            model_results[second_model],
        )

    summary = {
        "experiment_id": args.experiment_id,
        "subset_manifest_sha256": next(iter(manifest_hashes)),
        "prompt_version": next(iter(prompt_versions)),
        "prompt_sha256": next(iter(prompt_hashes)),
        "temperature_requested": next(iter(temperature_values)),
        "input_metadata": input_metadata,
        "within_model": within,
        "inter_model": inter,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tidy_rows = build_tidy_rows(summary)
    write_tidy_csv(args.output_csv, tidy_rows)

    print("Within-model mean pairwise agreement (exact label):")
    for model, values in within.items():
        exact_values = values["resolutions"]["exact_label"]
        print(
            f"  {model}: {exact_values['mean_pairwise_agreement']:.3f}; "
            f"Fleiss kappa={exact_values['fleiss_kappa']:.3f}; "
            f"unanimous={exact_values['unanimous_images']}/"
            f"{values['complete_images']}"
        )

    print("Inter-model matched-run agreement (exact label):")
    for comparison, values in inter.items():
        metric = values["resolutions"]["exact_label"]["matched_runs"]
        print(
            f"  {comparison}: {metric['agreement']:.3f}; "
            f"kappa={metric['cohen_kappa']:.3f}; n={metric['n']}"
        )

    print(f"JSON summary: {args.output_json.resolve()}")
    print(f"CSV summary: {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
