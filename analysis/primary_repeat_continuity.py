#!/usr/bin/env python3
"""Compare primary labels with modal labels from the three-run audit.

The script uses only the Python standard library. Run it from the public
replication bundle root after placing the primary and repeatability CSV files
at the paths defined in PRIMARY_FILES and REPEAT_FILES.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


NONE = "None of these / Not clearly moral in nature"
FOUNDATION = {
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
    NONE: "None",
}

PRIMARY_FILES = {
    "GPT-4o": Path("gpt4o_all_results.csv"),
    "GPT-5.6": Path("gpt5.6_all_results.csv"),
    "Qwen3.6 Flash": Path("qwen36_flash_all_results.csv"),
}
REPEAT_FILES = {
    "GPT-4o": Path("repeatability_results/gpt4o_subset_repeats.csv"),
    "GPT-5.6": Path("repeatability_results/gpt56_subset_repeats.csv"),
    "Qwen3.6 Flash": Path("repeatability_results/qwen36_subset_repeats.csv"),
}
EXPERIMENT_ID = "visual-moral-repeatability-v1"


def cohen_kappa(first: list[str], second: list[str]) -> float:
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    first_counts = Counter(first)
    second_counts = Counter(second)
    expected = sum(
        (first_counts[label] / n) * (second_counts[label] / n)
        for label in set(first_counts) | set(second_counts)
    )
    return 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1 - expected)


def load_primary(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {row["image_id"]: row["label"] for row in csv.DictReader(handle)}


def load_repeat_modes(path: Path) -> dict[str, str]:
    by_image: dict[str, list[str]] = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("experiment_id") == EXPERIMENT_ID and row.get("status") == "success":
                by_image[row["image_id"]].append(row["label"])

    modes: dict[str, str] = {}
    for image_id, labels in by_image.items():
        counts = Counter(labels)
        most_common = counts.most_common()
        if len(labels) != 3 or len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            raise ValueError(f"No unique three-run mode for {image_id}: {labels}")
        modes[image_id] = most_common[0][0]
    return modes


def main() -> None:
    transforms = {
        "exact_label": lambda label: label,
        "foundation": lambda label: FOUNDATION[label],
        "moral_vs_none": lambda label: "None" if label == NONE else "Moral label",
    }
    output: dict[str, object] = {}

    for model in PRIMARY_FILES:
        primary = load_primary(PRIMARY_FILES[model])
        modal = load_repeat_modes(REPEAT_FILES[model])
        image_ids = sorted(modal)
        if len(image_ids) != 32 or not set(image_ids) <= set(primary):
            raise ValueError(f"Unexpected subset for {model}")

        results = {}
        for resolution, transform in transforms.items():
            first = [transform(primary[image_id]) for image_id in image_ids]
            second = [transform(modal[image_id]) for image_id in image_ids]
            matches = sum(a == b for a, b in zip(first, second))
            results[resolution] = {
                "matches": matches,
                "n": len(image_ids),
                "agreement": round(matches / len(image_ids), 6),
                "cohen_kappa": round(cohen_kappa(first, second), 6),
            }

        output[model] = {
            "resolutions": results,
            "exact_label_changes": [
                {
                    "image_id": image_id,
                    "primary": primary[image_id],
                    "repeat_mode": modal[image_id],
                }
                for image_id in image_ids
                if primary[image_id] != modal[image_id]
            ],
        }

    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
