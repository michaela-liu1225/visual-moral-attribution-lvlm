#!/usr/bin/env python3
"""Recompute the descriptive results reported in thesis Chapter 4.

The script treats the three canonical model CSV files and the filtered
Qualtrics response export as the numerical sources of record. It does not
read model values from the derived Excel or Word reports.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

LABELS = [
    "Care",
    "Harm",
    "Fairness",
    "Cheating",
    "Loyalty",
    "Betrayal",
    "Respect / Authority",
    "Subversion",
    "Sanctity",
    "Degradation",
    "None of these / Not clearly moral in nature",
]
NONE_LABEL = LABELS[-1]

MODEL_FILES = {
    "GPT-4o": "gpt4o_all_results.csv",
    "GPT-5.6": "gpt5.6_all_results.csv",
    "Qwen3.6 Flash": "qwen36_flash_all_results.csv",
}

SURVEY_IMAGES = [
    "1-1",
    "9-2",
    "7-2",
    "37-3",
    "15-3",
    "g-1",
    "5-1",
    "3-1",
    "o-2",
    "g-2",
    "2-1",
    "25-2",
    "34-4",
    "o-1",
    "16-2",
    "35-3",
]

PART2_ITEMS = {
    "Individual": ["g-4_1", "g-3_1", "g-5_1"],
    "Intergroup": ["g-7_1", "g-6_1", "o-3_1"],
    "Institutional": ["g-8_1", "g-2_1", "g-1_1"],
}

VARIANT_TO_BASE = {
    "test16-1.jpg": ("16-1", "face occlusion"),
    "test16-2.jpg": ("16-2", "face occlusion"),
    "16-1upset.jpg": ("16-1", "distressed-expression edit"),
    "16-2happy.jpg": ("16-2", "positive-expression edit"),
    "1-1face.jpg": ("1-1", "face occlusion"),
    "1-1arm.jpg": ("1-1", "contact-region occlusion"),
    "17-2body.jpg": ("7-2", "body/pose occlusion"),
    "34-1text.jpg": ("34-1", "text ablation"),
    "g-7women.jpg": ("g-7", "participant-region occlusion"),
    "12-1hand.jpg": ("12-1", "ritual-gesture occlusion"),
    "32-2bg.jpg": ("32-2", "background replacement"),
}


def round_or_none(value: float | None, digits: int = 4):
    return None if value is None else round(float(value), digits)


def numeric_summary(values: list[float]) -> dict:
    # Keep NumPy optional for callers that only reuse the survey filtering and
    # aggregation functions (for example the prompt experiment's anonymiser).
    import numpy as np

    array = np.asarray(values, dtype=float)
    q1, q3 = np.percentile(array, [25, 75], method="linear")
    return {
        "n": len(values),
        "mean": round_or_none(array.mean()),
        "sd": round_or_none(statistics.stdev(values) if len(values) > 1 else 0.0),
        "median": round_or_none(np.median(array)),
        "q1": round_or_none(q1),
        "q3": round_or_none(q3),
        "iqr": round_or_none(q3 - q1),
        "min": round_or_none(array.min()),
        "max": round_or_none(array.max()),
    }


def load_model_rows(data_dir: Path) -> dict[str, dict[str, dict]]:
    result = {}
    expected_ids = None
    for model, filename in MODEL_FILES.items():
        path = data_dir / filename
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 127, (path, len(rows))
        assert len({row["image_id"] for row in rows}) == 127
        assert all(row["status"] == "success" for row in rows)
        assert all(row["label"] in LABELS for row in rows)
        for row in rows:
            row["confidence"] = int(row["confidence"])
            assert 0 <= row["confidence"] <= 100
        indexed = {row["image_id"]: row for row in rows}
        if expected_ids is None:
            expected_ids = set(indexed)
        else:
            assert set(indexed) == expected_ids
        result[model] = indexed
    return result


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    assert len(labels_a) == len(labels_b)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / len(labels_a)
    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    expected = sum(
        (count_a[label] / len(labels_a)) * (count_b[label] / len(labels_b))
        for label in LABELS
    )
    return (observed - expected) / (1 - expected)


def analyse_models(model_rows: dict[str, dict[str, dict]]) -> dict:
    models = list(MODEL_FILES)
    ids = list(model_rows[models[0]])
    distribution = {}
    confidence = {}
    for model in models:
        labels = [model_rows[model][image_id]["label"] for image_id in ids]
        scores = [model_rows[model][image_id]["confidence"] for image_id in ids]
        counts = Counter(labels)
        distribution[model] = {
            label: {"count": counts[label], "proportion": round(counts[label] / 127, 6)}
            for label in LABELS
        }
        confidence[model] = numeric_summary(scores)
        confidence[model]["moral_count"] = 127 - counts[NONE_LABEL]
        confidence[model]["moral_rate"] = round((127 - counts[NONE_LABEL]) / 127, 6)

    all_equal = {
        image_id: len({model_rows[m][image_id]["label"] for m in models}) == 1
        for image_id in ids
    }
    all_binary_equal = {
        image_id: len(
            {
                model_rows[m][image_id]["label"] != NONE_LABEL
                for m in models
            }
        )
        == 1
        for image_id in ids
    }

    confidence_by_agreement = {}
    for model in models:
        unanimous = [
            model_rows[model][i]["confidence"] for i in ids if all_equal[i]
        ]
        disagreement = [
            model_rows[model][i]["confidence"] for i in ids if not all_equal[i]
        ]
        confidence_by_agreement[model] = {
            "unanimous": numeric_summary(unanimous),
            "disagreement": numeric_summary(disagreement),
        }

    pairwise = {}
    for index, first in enumerate(models):
        for second in models[index + 1 :]:
            labels_a = [model_rows[first][i]["label"] for i in ids]
            labels_b = [model_rows[second][i]["label"] for i in ids]
            exact_count = sum(a == b for a, b in zip(labels_a, labels_b))
            binary_a = [label != NONE_LABEL for label in labels_a]
            binary_b = [label != NONE_LABEL for label in labels_b]
            binary_count = sum(a == b for a, b in zip(binary_a, binary_b))
            disagreements = Counter(
                tuple(sorted((a, b)))
                for a, b in zip(labels_a, labels_b)
                if a != b
            )
            pairwise[f"{first} | {second}"] = {
                "exact_count": exact_count,
                "exact_rate": round(exact_count / 127, 6),
                "binary_count": binary_count,
                "binary_rate": round(binary_count / 127, 6),
                "kappa": round(cohen_kappa(labels_a, labels_b), 6),
                "top_disagreement_pairs": [
                    {"labels": list(pair), "count": count}
                    for pair, count in disagreements.most_common(8)
                ],
            }

    agreement_pattern = Counter()
    for image_id in ids:
        labels = [model_rows[m][image_id]["label"] for m in models]
        unique = len(set(labels))
        if unique == 1:
            agreement_pattern["all_three_same"] += 1
        elif unique == 2:
            agreement_pattern["two_same_one_different"] += 1
        else:
            agreement_pattern["all_three_different"] += 1

    return {
        "label_distribution": distribution,
        "confidence": confidence,
        "confidence_by_exact_unanimity": confidence_by_agreement,
        "pairwise_agreement": pairwise,
        "three_model_exact_unanimity": {
            "count": sum(all_equal.values()),
            "rate": round(sum(all_equal.values()) / 127, 6),
        },
        "three_model_binary_unanimity": {
            "count": sum(all_binary_equal.values()),
            "rate": round(sum(all_binary_equal.values()) / 127, 6),
        },
        "agreement_pattern": dict(agreement_pattern),
    }


def load_filtered_survey(survey_zip: Path) -> list[dict]:
    with ZipFile(survey_zip) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        assert len(csv_names) == 1
        text = archive.read(csv_names[0]).decode("utf-8-sig")
    rows = list(csv.DictReader(io.StringIO(text)))
    retained = [
        row
        for row in rows
        if row.get("Status") == "IP Address"
        and row.get("Finished") == "True"
        and row.get("Progress") == "100"
    ]
    assert len(retained) == 56
    assert all(sum(bool(row[image].strip()) for image in SURVEY_IMAGES) == 5 for row in retained)
    return retained


def parse_survey_selection(answer: str) -> set[str]:
    selected = set()
    for label in LABELS:
        if f"{label} (" in answer:
            selected.add(label)
    if answer and not selected:
        raise ValueError(f"Unparsed survey response: {answer!r}")
    return selected


def survey_counts(rows: list[dict], remove_mixed_none: bool = False):
    counts = {image: Counter() for image in SURVEY_IMAGES}
    denominators = Counter()
    response_selection_counts = Counter()
    mixed_none_count = 0
    for row in rows:
        for image in SURVEY_IMAGES:
            answer = row[image].strip()
            if not answer:
                continue
            denominators[image] += 1
            selected = parse_survey_selection(answer)
            if NONE_LABEL in selected and len(selected) > 1:
                mixed_none_count += 1
                if remove_mixed_none:
                    selected.remove(NONE_LABEL)
            response_selection_counts[len(selected)] += 1
            counts[image].update(selected)
    assert sum(denominators.values()) == 280
    return counts, denominators, response_selection_counts, mixed_none_count


def analyse_survey(rows: list[dict], model_rows: dict[str, dict[str, dict]]) -> dict:
    counts, denominators, selection_sizes, mixed_none_count = survey_counts(rows)
    sensitive_counts, _, _, sensitive_mixed = survey_counts(rows, remove_mixed_none=True)
    assert mixed_none_count == 3
    assert sensitive_mixed == 3

    by_image = {}
    for image in SURVEY_IMAGES:
        maximum = max(counts[image].values())
        plurality = sorted(label for label, count in counts[image].items() if count == maximum)
        by_image[image] = {
            "n": denominators[image],
            "endorsement_counts": {label: counts[image][label] for label in LABELS},
            "endorsement_rates": {
                label: round(counts[image][label] / denominators[image], 6)
                for label in LABELS
            },
            "plurality_set": plurality,
            "plurality_count": maximum,
        }

    model_support = {}
    model_support_sensitivity = {}
    for model in MODEL_FILES:
        numerators = []
        rates = []
        sensitive_numerators = []
        plurality_matches = 0
        image_rows = {}
        for image in SURVEY_IMAGES:
            label = model_rows[model][image]["label"]
            numerator = counts[image][label]
            sensitive_numerator = sensitive_counts[image][label]
            denominator = denominators[image]
            rate = numerator / denominator
            numerators.append(numerator)
            sensitive_numerators.append(sensitive_numerator)
            rates.append(rate)
            is_plurality = label in by_image[image]["plurality_set"]
            plurality_matches += is_plurality
            image_rows[image] = {
                "model_label": label,
                "endorsement_count": numerator,
                "n": denominator,
                "endorsement_rate": round(rate, 6),
                "plurality_match": is_plurality,
            }
        model_support[model] = {
            "macro": round(sum(rates) / 16, 6),
            "micro": round(sum(numerators) / 280, 6),
            "micro_numerator": sum(numerators),
            "plurality_correspondence_count": plurality_matches,
            "plurality_correspondence_rate": round(plurality_matches / 16, 6),
            "by_image": image_rows,
        }
        sensitive_rates = [
            sensitive_counts[image][model_rows[model][image]["label"]]
            / denominators[image]
            for image in SURVEY_IMAGES
        ]
        model_support_sensitivity[model] = {
            "macro": round(sum(sensitive_rates) / 16, 6),
            "micro": round(sum(sensitive_numerators) / 280, 6),
            "micro_numerator": sum(sensitive_numerators),
        }

    part2 = {}
    for group, items in PART2_ITEMS.items():
        part2[group] = {"items": {}, "numeric_total": 0}
        for item in items:
            values = [int(row[item]) for row in rows if row[item].strip()]
            part2[group]["items"][item] = numeric_summary(values)
            part2[group]["numeric_total"] += len(values)
        part2[group]["not_applicable_count"] = 112 - part2[group]["numeric_total"]

    return {
        "retained_responses": len(rows),
        "part1_total_image_responses": sum(denominators.values()),
        "part1_denominators": dict(denominators),
        "selection_count_distribution": dict(sorted(selection_sizes.items())),
        "mixed_none_with_moral_responses": mixed_none_count,
        "by_image": by_image,
        "model_support": model_support,
        "mixed_none_sensitivity": model_support_sensitivity,
        "part2": part2,
    }


def parse_variant_results(text_path: Path) -> dict[str, dict[str, dict]]:
    text = text_path.read_text(encoding="utf-8")
    start_4o = text.index("gpt4o") + len("gpt4o")
    start_56 = text.index("\ngpt5.6")
    start_qwen = text.index("\nqwen")
    sections = {
        "GPT-4o": text[start_4o:start_56],
        "GPT-5.6": text[start_56 + len("\ngpt5.6") : start_qwen],
        "Qwen3.6 Flash": text[start_qwen + len("\nqwen") :],
    }
    pattern = re.compile(
        r"Image:\s*([^\n]+)\nLabel:\s*([^\n]+)\nConfidence:\s*(\d+)",
        re.MULTILINE,
    )
    parsed = {}
    for model, section in sections.items():
        rows = {}
        for filename, label, confidence in pattern.findall(section):
            assert filename in VARIANT_TO_BASE
            assert label in LABELS
            rows[filename] = {"label": label, "confidence": int(confidence)}
        parsed[model] = rows
    assert len(parsed["GPT-4o"]) == 11
    assert len(parsed["GPT-5.6"]) == 11
    assert len(parsed["Qwen3.6 Flash"]) == 10
    assert "16-1upset.jpg" not in parsed["Qwen3.6 Flash"]
    return parsed


def analyse_variants(
    variant_rows: dict[str, dict[str, dict]],
    model_rows: dict[str, dict[str, dict]],
) -> dict:
    by_model = {}
    by_variant = defaultdict(dict)
    for model in MODEL_FILES:
        comparisons = []
        for variant, (base, intervention) in VARIANT_TO_BASE.items():
            if variant not in variant_rows[model]:
                by_variant[variant][model] = {
                    "base": base,
                    "intervention": intervention,
                    "status": "failed",
                }
                continue
            modified = variant_rows[model][variant]
            baseline = model_rows[model][base]
            record = {
                "base": base,
                "intervention": intervention,
                "status": "success",
                "base_label": baseline["label"],
                "modified_label": modified["label"],
                "base_confidence": baseline["confidence"],
                "modified_confidence": modified["confidence"],
                "label_changed": modified["label"] != baseline["label"],
                "confidence_change": modified["confidence"] - baseline["confidence"],
            }
            comparisons.append(record)
            by_variant[variant][model] = record
        changes = sum(record["label_changed"] for record in comparisons)
        deltas = [record["confidence_change"] for record in comparisons]
        by_model[model] = {
            "valid_comparisons": len(comparisons),
            "label_changes": changes,
            "label_change_rate": round(changes / len(comparisons), 6),
            "confidence_change": numeric_summary(deltas),
        }
    return {
        "by_model": by_model,
        "by_variant": dict(by_variant),
        "valid_comparisons_total": sum(
            summary["valid_comparisons"] for summary in by_model.values()
        ),
        "label_changes_total": sum(
            summary["label_changes"] for summary in by_model.values()
        ),
        "failed_comparisons": 1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--survey-zip",
        type=Path,
        required=True,
        help=(
            "Path to the restricted raw Qualtrics response ZIP. "
            "Do not place the unredacted export in a public replication bundle."
        ),
    )
    parser.add_argument(
        "--variant-text",
        type=Path,
        default=Path(__file__).with_name("variant_console_transcript.txt"),
    )
    args = parser.parse_args()

    model_rows = load_model_rows(args.data_dir)
    survey_rows = load_filtered_survey(args.survey_zip)
    variant_rows = parse_variant_results(args.variant_text)
    output = {
        "models": analyse_models(model_rows),
        "survey": analyse_survey(survey_rows, model_rows),
        "variants": analyse_variants(variant_rows, model_rows),
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
