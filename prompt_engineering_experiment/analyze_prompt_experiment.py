#!/usr/bin/env python3
"""Analyse the frozen P0/P1/P2 prompt experiment without participant PII.

The inferential unit is the image. Repeated model calls are first collapsed
within each model/condition/image cell; they are never treated as independent
observations in confidence intervals or hypothesis tests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from repeatability_subset_common import VALID_LABELS  # noqa: E402


CONDITIONS = ("p0", "p1", "p2")
RESULT_SCHEMA_VERSION = "prompt-results-v1"
COMPARISONS = (
    ("p2_minus_p0", "p0", "p2"),
    ("p1_minus_p0", "p0", "p1"),
    ("p2_minus_p1", "p1", "p2"),
)
DEFAULT_BOOTSTRAP_SEED = 20260905
DEFAULT_BOOTSTRAP_REPLICATES = 20_000
EPSILON = 1e-12


class AnalysisInputError(ValueError):
    """Raised when an input file violates the frozen analysis contract."""


class IncompleteExperimentError(AnalysisInputError):
    """Raised when required successful trial cells are incomplete."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Match the runner's canonical JSON hash exactly."""

    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_prompt_snapshot(path: Path) -> str:
    """Hash the prompt text using the runner/freeze trailing-newline convention."""

    text = path.read_text(encoding="utf-8").rstrip("\n")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AnalysisInputError(f"Expected a JSON object in {path.name}.")
    return value


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise AnalysisInputError(f"CSV has no header: {path.name}")
        return list(reader.fieldnames), list(reader)


def first_field(
    fields: Iterable[str], candidates: Sequence[str], *, required: bool = True
) -> str | None:
    available = set(fields)
    for candidate in candidates:
        if candidate in available:
            return candidate
    if required:
        raise AnalysisInputError(
            "Missing required column. Expected one of: " + ", ".join(candidates)
        )
    return None


def parse_float(value: str, *, field: str, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise AnalysisInputError(
            f"Invalid numeric {field} for {context}: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise AnalysisInputError(f"Non-finite {field} for {context}: {value!r}")
    return result


def parse_int(value: str, *, field: str, context: str) -> int:
    number = parse_float(value, field=field, context=context)
    if not number.is_integer():
        raise AnalysisInputError(
            f"Expected integer {field} for {context}; got {value!r}."
        )
    return int(number)


def parse_bool(value: str) -> bool:
    normalised = value.strip().lower()
    if normalised in {"1", "true", "yes", "y"}:
        return True
    if normalised in {"0", "false", "no", "n", ""}:
        return False
    raise AnalysisInputError(f"Cannot parse boolean value {value!r}.")


def mean_or_none(values: Iterable[float | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return statistics.fmean(usable) if usable else None


def weighted_mean_or_none(
    values_and_weights: Iterable[tuple[float | None, int]]
) -> float | None:
    usable = [
        (float(value), int(weight))
        for value, weight in values_and_weights
        if value is not None and int(weight) > 0
    ]
    if not usable:
        return None
    return sum(value * weight for value, weight in usable) / sum(
        weight for _, weight in usable
    )


def round_or_none(value: float | None, digits: int = 9) -> float | None:
    return None if value is None else round(float(value), digits)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("Cannot calculate a quantile of an empty sequence.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def exact_sign_flip_p(deltas: Sequence[float]) -> float | None:
    """Two-sided exact randomisation p-value for the image-level mean delta."""

    if not deltas:
        return None
    if len(deltas) > 20:
        raise AnalysisInputError(
            "Exact sign-flip enumeration is limited to 20 paired images."
        )
    observed = abs(statistics.fmean(deltas))
    extreme = 0
    total = 1 << len(deltas)
    for signs in itertools.product((-1.0, 1.0), repeat=len(deltas)):
        permuted = abs(
            sum(sign * delta for sign, delta in zip(signs, deltas)) / len(deltas)
        )
        if permuted + EPSILON >= observed:
            extreme += 1
    return extreme / total


def bootstrap_mean_ci(
    deltas: Sequence[float], *, seed: int, replicates: int
) -> tuple[float | None, float | None]:
    """Percentile CI from resampling paired images, not repeated calls."""

    if not deltas:
        return None, None
    if replicates < 1:
        raise AnalysisInputError("Bootstrap replicates must be positive.")
    rng = random.Random(seed)
    n_images = len(deltas)
    samples = [
        sum(deltas[rng.randrange(n_images)] for _ in range(n_images)) / n_images
        for _ in range(replicates)
    ]
    return quantile(samples, 0.025), quantile(samples, 0.975)


def exact_mcnemar_p(wrong_to_right: int, right_to_wrong: int) -> float:
    """Two-sided exact McNemar/binomial p-value for discordant image pairs."""

    discordant = wrong_to_right + right_to_wrong
    if discordant == 0:
        return 1.0
    tail = min(wrong_to_right, right_to_wrong)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (
        2**discordant
    )
    return min(1.0, 2 * probability)


def modal_label(labels: Sequence[str]) -> tuple[str, int, list[str]]:
    if not labels:
        raise ValueError("Cannot choose a modal label from an empty sequence.")
    counts = Counter(labels)
    maximum = max(counts.values())
    tied = [label for label in VALID_LABELS if counts[label] == maximum]
    return tied[0], maximum, tied


def pairwise_label_agreement(labels: Sequence[str]) -> float | None:
    if len(labels) < 2:
        return None
    pairs = list(itertools.combinations(labels, 2))
    return sum(first == second for first, second in pairs) / len(pairs)


def load_evaluation_images(path: Path) -> list[str]:
    fields, rows = read_csv(path)
    image_field = first_field(fields, ("image_id", "image"))
    assert image_field is not None
    image_ids = [row[image_field].strip() for row in rows]
    if not image_ids or any(not image_id for image_id in image_ids):
        raise AnalysisInputError("Evaluation manifest contains a blank image ID.")
    if len(image_ids) != len(set(image_ids)):
        raise AnalysisInputError("Evaluation manifest contains duplicate image IDs.")
    return image_ids


def load_frozen_schedule(
    path: Path, *, expected_images: Sequence[str], repeats: int
) -> dict[tuple[int, str, str], dict[str, int | str]]:
    fields, rows = read_csv(path)
    required = (
        "sequence_index",
        "run_number",
        "image_block_position",
        "condition_position",
        "image_id",
        "condition",
    )
    missing_fields = [field for field in required if field not in fields]
    if missing_fields:
        raise AnalysisInputError(
            "Frozen schedule is missing columns: " + ", ".join(missing_fields)
        )

    schedule: dict[tuple[int, str, str], dict[str, int | str]] = {}
    sequence_indices: list[int] = []
    expected_image_set = set(expected_images)
    for row_number, row in enumerate(rows, start=2):
        image_id = row["image_id"].strip()
        condition = row["condition"].strip().lower()
        context = f"{path.name} row {row_number} ({image_id}/{condition})"
        run_number = parse_int(
            row["run_number"], field="run_number", context=context
        )
        sequence_index = parse_int(
            row["sequence_index"], field="sequence_index", context=context
        )
        image_block_position = parse_int(
            row["image_block_position"],
            field="image_block_position",
            context=context,
        )
        condition_position = parse_int(
            row["condition_position"],
            field="condition_position",
            context=context,
        )
        if image_id not in expected_image_set or condition not in CONDITIONS:
            raise AnalysisInputError(f"Out-of-scope slot in frozen {context}.")
        if not 1 <= run_number <= repeats:
            raise AnalysisInputError(f"Run number outside 1..{repeats} in {context}.")
        key = (run_number, image_id, condition)
        if key in schedule:
            raise AnalysisInputError(f"Duplicate slot in frozen {context}.")
        schedule[key] = {
            "sequence_index": sequence_index,
            "run_number": run_number,
            "image_block_position": image_block_position,
            "condition_position": condition_position,
            "image_id": image_id,
            "condition": condition,
        }
        sequence_indices.append(sequence_index)

    expected_keys = {
        (run_number, image_id, condition)
        for run_number in range(1, repeats + 1)
        for image_id in expected_images
        for condition in CONDITIONS
    }
    if set(schedule) != expected_keys:
        missing = sorted(expected_keys - set(schedule))
        extra = sorted(set(schedule) - expected_keys)
        raise AnalysisInputError(
            "Frozen schedule does not contain exactly the required slots: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    if sorted(sequence_indices) != list(range(1, len(expected_keys) + 1)):
        raise AnalysisInputError(
            "Frozen schedule sequence_index must be a permutation of 1.."
            f"{len(expected_keys)}."
        )
    return schedule


def validate_frozen_protocol(
    *,
    manifest: dict[str, Any],
    config: dict[str, Any],
    config_path: Path,
    evaluation_manifest_path: Path,
    schedule_path: Path,
    expected_images: Sequence[str],
) -> dict[str, Any]:
    """Validate the frozen inputs and return row-level expectations."""

    canonical_fields = (
        "config_sha256",
        "evaluation_manifest_sha256",
        "cue_discovery_manifest_sha256",
        "schedule_sha256",
        "runner_sha256",
        "evaluation_image_sha256",
        "prompts",
    )
    required_fields = ("protocol_id", "protocol_sha256", "schedule_seed", *canonical_fields)
    missing = [field for field in required_fields if field not in manifest]
    if missing:
        raise AnalysisInputError(
            "Frozen protocol manifest is missing fields: " + ", ".join(missing)
        )
    calculated_protocol_hash = canonical_hash(
        {field: manifest[field] for field in canonical_fields}
    )
    if calculated_protocol_hash != str(manifest["protocol_sha256"]):
        raise AnalysisInputError(
            "Frozen protocol manifest canonical hash does not match protocol_sha256."
        )

    current_hashes = {
        "config_sha256": sha256_file(config_path),
        "evaluation_manifest_sha256": sha256_file(evaluation_manifest_path),
        "schedule_sha256": sha256_file(schedule_path),
    }
    for field, current_hash in current_hashes.items():
        if current_hash != str(manifest[field]):
            raise AnalysisInputError(
                f"Current {field.removesuffix('_sha256')} no longer matches the "
                f"frozen protocol manifest ({current_hash} != {manifest[field]})."
            )
    if str(config.get("protocol_id", "")) != str(manifest["protocol_id"]):
        raise AnalysisInputError(
            "experiment_config.json protocol_id does not match the frozen manifest."
        )
    if str(config.get("schedule_seed", "")) != str(manifest["schedule_seed"]):
        raise AnalysisInputError(
            "experiment_config.json schedule_seed does not match the frozen manifest."
        )

    image_hashes = manifest["evaluation_image_sha256"]
    if not isinstance(image_hashes, dict) or set(image_hashes) != set(expected_images):
        raise AnalysisInputError(
            "Frozen evaluation_image_sha256 does not cover exactly the 16 evaluation images."
        )
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in image_hashes.values()
    ):
        raise AnalysisInputError("Frozen evaluation image hash is malformed.")

    raw_prompts = manifest["prompts"]
    if not isinstance(raw_prompts, list):
        raise AnalysisInputError("Frozen prompts entry must be a list.")
    prompts: dict[str, dict[str, str]] = {}
    for entry in raw_prompts:
        if not isinstance(entry, dict):
            raise AnalysisInputError("Every frozen prompt entry must be an object.")
        condition = str(entry.get("condition", "")).lower()
        if condition not in CONDITIONS or condition in prompts:
            raise AnalysisInputError(
                f"Invalid or duplicate prompt condition in frozen manifest: {condition!r}."
            )
        version = str(entry.get("version", ""))
        prompt_hash = str(entry.get("sha256", ""))
        filename = str(entry.get("filename", ""))
        if not version or len(prompt_hash) != 64 or not filename:
            raise AnalysisInputError(
                f"Incomplete frozen prompt metadata for condition {condition}."
            )
        if Path(filename).name != filename:
            raise AnalysisInputError(
                f"Frozen prompt filename must be a basename: {filename!r}."
            )
        prompt_path = EXPERIMENT_DIR / "prompts" / filename
        if not prompt_path.is_file():
            raise AnalysisInputError(f"Frozen prompt snapshot is missing: {filename}.")
        if sha256_prompt_snapshot(prompt_path) != prompt_hash:
            raise AnalysisInputError(
                f"Prompt snapshot {filename} no longer matches its frozen hash."
            )
        prompts[condition] = {
            "condition": condition,
            "version": version,
            "sha256": prompt_hash,
            "filename": filename,
        }
    if set(prompts) != set(CONDITIONS):
        raise AnalysisInputError("Frozen manifest must contain exactly P0, P1, and P2.")

    model_specs = config.get("model_specs")
    if not isinstance(model_specs, dict) or not model_specs:
        raise AnalysisInputError("experiment_config.json model_specs must be non-empty.")
    for model_key, spec in model_specs.items():
        if not isinstance(spec, dict) or not spec.get("provider") or not spec.get(
            "requested_model"
        ):
            raise AnalysisInputError(
                f"Incomplete provider/requested_model config for model {model_key}."
            )

    return {
        "protocol_id": str(manifest["protocol_id"]),
        "protocol_sha256": str(manifest["protocol_sha256"]),
        "schedule_seed": str(manifest["schedule_seed"]),
        "schedule_sha256": str(manifest["schedule_sha256"]),
        "evaluation_manifest_sha256": str(
            manifest["evaluation_manifest_sha256"]
        ),
        "image_hashes": {str(key): str(value) for key, value in image_hashes.items()},
        "prompts": prompts,
        "model_specs": model_specs,
        "temperature": float(config.get("temperature", 0)),
    }


def load_human_targets(
    path: Path, evaluation_images: Sequence[str]
) -> dict[str, Any]:
    fields, rows = read_csv(path)
    image_field = first_field(fields, ("image_id", "image"))
    label_field = first_field(fields, ("label", "category"))
    rate_field = first_field(
        fields, ("endorsement_rate", "human_endorsement_rate", "rate")
    )
    count_field = first_field(
        fields, ("endorsement_count", "human_endorsement_count", "count"), required=False
    )
    n_field = first_field(
        fields, ("n_responses", "n", "denominator"), required=False
    )
    plurality_field = first_field(
        fields, ("is_plurality", "plurality_match"), required=False
    )
    sensitivity_rate_field = first_field(
        fields,
        (
            "mixed_none_removed_rate",
            "endorsement_rate_mixed_none_removed",
            "rate_without_mixed_none",
            "sensitivity_endorsement_rate",
        ),
        required=False,
    )
    sensitivity_count_field = first_field(
        fields,
        (
            "mixed_none_removed_count",
            "endorsement_count_mixed_none_removed",
            "count_without_mixed_none",
        ),
        required=False,
    )
    assert image_field is not None and label_field is not None and rate_field is not None

    base_rates: dict[tuple[str, str], float] = {}
    sensitivity_rates: dict[tuple[str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    denominators: dict[str, int] = {}
    recorded_plurality: dict[tuple[str, str], bool] = {}

    for row_number, row in enumerate(rows, start=2):
        image_id = row[image_field].strip()
        label = row[label_field].strip()
        context = f"{path.name} row {row_number} ({image_id}/{label})"
        if image_id not in evaluation_images:
            raise AnalysisInputError(f"Unexpected image ID in {context}.")
        if label not in VALID_LABELS:
            raise AnalysisInputError(f"Unexpected label in {context}: {label!r}")
        key = (image_id, label)
        if key in base_rates:
            raise AnalysisInputError(f"Duplicate human target row for {image_id}/{label}.")
        rate = parse_float(row[rate_field], field=rate_field, context=context)
        if not 0 <= rate <= 1:
            raise AnalysisInputError(f"Endorsement rate outside [0, 1] in {context}.")
        base_rates[key] = rate

        if n_field and row[n_field].strip():
            denominator = parse_int(row[n_field], field=n_field, context=context)
            if denominator <= 0:
                raise AnalysisInputError(f"Non-positive denominator in {context}.")
            previous = denominators.setdefault(image_id, denominator)
            if previous != denominator:
                raise AnalysisInputError(
                    f"Inconsistent denominator for image {image_id}: "
                    f"{previous} versus {denominator}."
                )
        if count_field and row[count_field].strip():
            count = parse_int(row[count_field], field=count_field, context=context)
            counts[key] = count
            if image_id in denominators:
                expected_rate = count / denominators[image_id]
                if abs(expected_rate - rate) > 1.1e-6:
                    raise AnalysisInputError(
                        f"Count/rate mismatch in {context}: {count}/"
                        f"{denominators[image_id]} != {rate}."
                    )
        if plurality_field:
            recorded_plurality[key] = parse_bool(row[plurality_field])

        sensitivity_value: float | None = None
        sensitivity_count: int | None = None
        if sensitivity_count_field and row[sensitivity_count_field].strip():
            sensitivity_count = parse_int(
                row[sensitivity_count_field],
                field=sensitivity_count_field,
                context=context,
            )
            if image_id not in denominators:
                raise AnalysisInputError(
                    f"Sensitivity count has no denominator in {context}."
                )
            if not 0 <= sensitivity_count <= denominators[image_id]:
                raise AnalysisInputError(
                    f"Sensitivity count outside 0..n in {context}."
                )
        if sensitivity_rate_field and row[sensitivity_rate_field].strip():
            sensitivity_value = parse_float(
                row[sensitivity_rate_field],
                field=sensitivity_rate_field,
                context=context,
            )
            if sensitivity_count is not None:
                expected_sensitivity_rate = (
                    sensitivity_count / denominators[image_id]
                )
                if abs(expected_sensitivity_rate - sensitivity_value) > 1.1e-6:
                    raise AnalysisInputError(
                        f"Sensitivity count/rate mismatch in {context}: "
                        f"{sensitivity_count}/{denominators[image_id]} != "
                        f"{sensitivity_value}."
                    )
        elif sensitivity_count is not None:
            sensitivity_value = sensitivity_count / denominators[image_id]
        if sensitivity_value is not None:
            if not 0 <= sensitivity_value <= 1:
                raise AnalysisInputError(
                    f"Sensitivity rate outside [0, 1] in {context}."
                )
            sensitivity_rates[key] = sensitivity_value

    expected_keys = {
        (image_id, label) for image_id in evaluation_images for label in VALID_LABELS
    }
    missing = sorted(expected_keys - set(base_rates))
    extra = sorted(set(base_rates) - expected_keys)
    if missing or extra:
        raise AnalysisInputError(
            f"Human target matrix is not complete 16 x {len(VALID_LABELS)}: "
            f"missing={missing[:5]}, extra={extra[:5]}."
        )
    if sensitivity_rates and set(sensitivity_rates) != expected_keys:
        raise AnalysisInputError(
            "Mixed-None sensitivity data are only partially populated."
        )
    missing_denominators = sorted(set(evaluation_images) - set(denominators))
    if missing_denominators:
        raise AnalysisInputError(
            "Human targets must provide n_responses for every image so the "
            "pre-specified micro summary can be calculated; missing="
            f"{missing_denominators}."
        )

    def derive_plurality(rate_map: dict[tuple[str, str], float]) -> dict[str, list[str]]:
        result: dict[str, list[str]] = {}
        for image_id in evaluation_images:
            if all((image_id, label) in counts for label in VALID_LABELS) and rate_map is base_rates:
                maximum = max(counts[(image_id, label)] for label in VALID_LABELS)
                result[image_id] = [
                    label
                    for label in VALID_LABELS
                    if counts[(image_id, label)] == maximum
                ]
            else:
                maximum_rate = max(rate_map[(image_id, label)] for label in VALID_LABELS)
                result[image_id] = [
                    label
                    for label in VALID_LABELS
                    if abs(rate_map[(image_id, label)] - maximum_rate) <= EPSILON
                ]
        return result

    plurality = derive_plurality(base_rates)
    if recorded_plurality:
        for image_id in evaluation_images:
            recorded = [
                label
                for label in VALID_LABELS
                if recorded_plurality[(image_id, label)]
            ]
            if recorded != plurality[image_id]:
                raise AnalysisInputError(
                    f"Recorded plurality flags disagree with counts for {image_id}: "
                    f"{recorded} versus {plurality[image_id]}."
                )
    sensitivity_plurality = (
        derive_plurality(sensitivity_rates) if sensitivity_rates else None
    )
    return {
        "base_rates": base_rates,
        "sensitivity_rates": sensitivity_rates or None,
        "n_by_image": denominators,
        "plurality": plurality,
        "sensitivity_plurality": sensitivity_plurality,
        "row_count": len(rows),
        "sha256": sha256_file(path),
        "filename": path.name,
    }


def discover_trial_files(results_root: Path, explicit: Sequence[Path]) -> list[Path]:
    if explicit:
        files = [path.resolve() for path in explicit]
    else:
        files = sorted(path.resolve() for path in results_root.glob("*/trials.csv"))
        if not files:
            combined = results_root / "prompt_experiment_results.csv"
            root_trials = results_root / "trials.csv"
            files = [
                path.resolve() for path in (combined, root_trials) if path.is_file()
            ]
    if not files:
        raise FileNotFoundError(
            "No trial CSV found. Expected results/<model>/trials.csv or "
            "results/prompt_experiment_results.csv."
        )
    missing = [path for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Trial CSV not found: " + ", ".join(map(str, missing)))
    if len(files) != len(set(files)):
        raise AnalysisInputError("The same trial CSV was supplied more than once.")
    return files


def load_trial_rows(paths: Sequence[Path]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    sources: list[dict[str, str]] = []
    for path in paths:
        fields, source_rows = read_csv(path)
        for required_candidates in (
            ("condition", "prompt_condition"),
            ("image_id", "image"),
            ("label", "predicted_label"),
            ("confidence", "model_confidence"),
            ("run_number", "repeat", "repeat_index", "run_id"),
        ):
            first_field(fields, required_candidates)
        inferred_model = path.parent.name if path.name == "trials.csv" else ""
        for source_row_number, row in enumerate(source_rows, start=2):
            row["__source_filename"] = path.name
            row["__source_row_number"] = str(source_row_number)
            row["__inferred_model_key"] = inferred_model
            rows.append(row)
        sources.append(
            {
                "filename": path.name,
                "parent_directory": path.parent.name,
                "sha256": sha256_file(path),
                "rows": str(len(source_rows)),
            }
        )
    return rows, sources


def row_value(row: dict[str, str], candidates: Sequence[str], default: str = "") -> str:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return str(row[candidate]).strip()
    return default


def choose_protocol_hash(
    explicit_hash: str | None, manifest: dict[str, Any]
) -> str:
    frozen_hash = str(manifest.get("protocol_sha256", ""))
    if not frozen_hash:
        raise AnalysisInputError("Frozen protocol manifest has no protocol_sha256.")
    if explicit_hash and explicit_hash != frozen_hash:
        raise AnalysisInputError(
            "--protocol-sha256 does not match the frozen protocol manifest: "
            f"{explicit_hash} != {frozen_hash}."
        )
    return explicit_hash or frozen_hash


def resolve_models(
    config: dict[str, Any], selected_models: Sequence[str] | None
) -> list[str]:
    model_specs = config.get("model_specs")
    if not isinstance(model_specs, dict) or not model_specs:
        raise AnalysisInputError("experiment_config.json model_specs must be non-empty.")
    configured = list(model_specs)
    if selected_models is None:
        return configured
    if not selected_models:
        raise AnalysisInputError("--models must name at least one configured model.")
    if len(selected_models) != len(set(selected_models)):
        raise AnalysisInputError("--models contains a duplicate model key.")
    unknown = [model for model in selected_models if model not in model_specs]
    if unknown:
        raise AnalysisInputError(
            "Unknown --models key(s); expected keys from model_specs: "
            + ", ".join(unknown)
        )
    return list(selected_models)


def require_row_equal(
    row: dict[str, str], *, field: str, expected: str, context: str
) -> str:
    actual = row_value(row, (field,))
    if actual != expected:
        raise AnalysisInputError(
            f"Frozen metadata mismatch for {field} in {context}: "
            f"{actual!r} != {expected!r}."
        )
    return actual


def successful_trial_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    """Critical fields that must match before a copied row may be ignored."""

    return (
        row["model_key"],
        row["condition"],
        row["image_id"],
        row["run_number"],
        row["label"],
        row["confidence"],
        row["protocol_sha256"],
        row["prompt_version"],
        row["prompt_sha256"],
        row["image_sha256"],
        row["provider"],
        row["requested_model"],
        row["actual_model"],
    )


def normalise_trials(
    rows: Sequence[dict[str, str]],
    *,
    selected_protocol_hash: str,
    expected_images: Sequence[str],
    repeats: int,
    expected_models: Sequence[str],
    frozen: dict[str, Any],
    schedule: dict[tuple[int, str, str], dict[str, int | str]],
) -> dict[str, Any]:
    expected_image_set = set(expected_images)
    selected_model_set = set(expected_models)
    normalised: list[dict[str, Any]] = []
    ignored_scope_rows = 0
    successful_rows_validated = 0

    for row in rows:
        model_key = row_value(row, ("model_key", "model")) or row["__inferred_model_key"]
        condition = row_value(row, ("condition", "prompt_condition")).lower()
        image_id = row_value(row, ("image_id", "image"))
        looks_like_planned_slot = (
            condition in CONDITIONS and image_id in expected_image_set
        )
        if not model_key and looks_like_planned_slot:
            raise AnalysisInputError(
                f"Missing model_key in {row['__source_filename']} row "
                f"{row['__source_row_number']}."
            )
        if model_key not in selected_model_set:
            ignored_scope_rows += 1
            continue
        if condition not in CONDITIONS or image_id not in expected_image_set:
            raise AnalysisInputError(
                f"Selected-model row is not a frozen schedule slot: "
                f"{row['__source_filename']} row {row['__source_row_number']} "
                f"({model_key}/{condition}/{image_id})."
            )
        context = (
            f"{row['__source_filename']} row {row['__source_row_number']} "
            f"({model_key}/{condition}/{image_id})"
        )
        protocol_hash = row_value(row, ("protocol_sha256", "protocol_hash"))
        if protocol_hash != selected_protocol_hash:
            raise AnalysisInputError(
                f"Protocol hash mismatch in in-scope {context}: "
                f"{protocol_hash!r} != {selected_protocol_hash!r}. Do not mix "
                "protocol versions in an analysis input."
            )
        run_number = parse_int(
            row_value(row, ("run_number", "repeat", "repeat_index", "run_id")),
            field="run_number",
            context=context,
        )
        if not 1 <= run_number <= repeats:
            raise AnalysisInputError(
                f"Run number outside 1..{repeats} in {context}: {run_number}."
            )
        schedule_key = (run_number, image_id, condition)
        if schedule_key not in schedule:
            raise AnalysisInputError(f"No frozen schedule slot exists for {context}.")
        status = row_value(row, ("status", "trial_status"), default="success").lower()
        if status not in {"success", "failure"}:
            raise AnalysisInputError(
                f"Unexpected trial status in {context}: {status!r}."
            )
        item: dict[str, Any] = {
            "model_key": model_key,
            "condition": condition,
            "image_id": image_id,
            "run_number": run_number,
            "status": status,
            "trial_id": row_value(row, ("trial_id",)),
            "protocol_sha256": protocol_hash,
            "actual_model": row_value(row, ("actual_model", "model_version")),
            "source": row["__source_filename"],
            "source_row": int(row["__source_row_number"]),
        }
        if status == "success":
            prompt = frozen["prompts"][condition]
            model_spec = frozen["model_specs"][model_key]
            schedule_row = schedule[schedule_key]
            metadata_expectations = {
                "schema_version": RESULT_SCHEMA_VERSION,
                "protocol_id": frozen["protocol_id"],
                "protocol_sha256": selected_protocol_hash,
                "schedule_seed": frozen["schedule_seed"],
                "schedule_sha256": frozen["schedule_sha256"],
                "prompt_version": prompt["version"],
                "prompt_sha256": prompt["sha256"],
                "evaluation_manifest_sha256": frozen[
                    "evaluation_manifest_sha256"
                ],
                "image_sha256": frozen["image_hashes"][image_id],
                "model_key": model_key,
                "provider": str(model_spec["provider"]),
                "requested_model": str(model_spec["requested_model"]),
            }
            validated_metadata = {
                field: require_row_equal(
                    row, field=field, expected=expected, context=context
                )
                for field, expected in metadata_expectations.items()
            }
            temperature = parse_float(
                row_value(row, ("temperature_requested",)),
                field="temperature_requested",
                context=context,
            )
            if abs(temperature - frozen["temperature"]) > EPSILON:
                raise AnalysisInputError(
                    f"Frozen temperature mismatch in {context}: "
                    f"{temperature} != {frozen['temperature']}."
                )
            reasoning_text = row_value(row, ("reasoning_setting",))
            try:
                reasoning_setting = json.loads(reasoning_text)
            except json.JSONDecodeError as error:
                raise AnalysisInputError(
                    f"Invalid reasoning_setting JSON in {context}: "
                    f"{reasoning_text!r}."
                ) from error
            if reasoning_setting != model_spec.get("reasoning", {}):
                raise AnalysisInputError(
                    f"Frozen reasoning_setting mismatch in {context}: "
                    f"{reasoning_setting!r} != {model_spec.get('reasoning', {})!r}."
                )
            require_row_equal(
                row,
                field="image_detail",
                expected=str(model_spec["image_detail"]),
                context=context,
            )
            for field in (
                "sequence_index",
                "image_block_position",
                "condition_position",
            ):
                actual = parse_int(
                    row_value(row, (field,)), field=field, context=context
                )
                expected = int(schedule_row[field])
                if actual != expected:
                    raise AnalysisInputError(
                        f"Frozen schedule mismatch for {field} in {context}: "
                        f"{actual} != {expected}."
                    )

            actual_model = row_value(row, ("actual_model", "model_version"))
            if not actual_model:
                raise AnalysisInputError(
                    f"Successful trial has blank actual_model in {context}."
                )
            expected_trial_id = canonical_hash(
                {
                    "protocol_sha256": selected_protocol_hash,
                    "model_key": model_key,
                    "provider": str(model_spec["provider"]),
                    "requested_model": str(model_spec["requested_model"]),
                    "condition": condition,
                    "prompt_sha256": prompt["sha256"],
                    "run_number": run_number,
                    "image_id": image_id,
                }
            )
            trial_id_value = require_row_equal(
                row, field="trial_id", expected=expected_trial_id, context=context
            )
            label = row_value(row, ("label", "predicted_label"))
            if label not in VALID_LABELS:
                raise AnalysisInputError(f"Invalid successful label in {context}: {label!r}")
            confidence = parse_float(
                row_value(row, ("confidence", "model_confidence")),
                field="confidence",
                context=context,
            )
            if not 0 <= confidence <= 100:
                raise AnalysisInputError(f"Confidence outside [0, 100] in {context}.")
            item.update(
                {
                    "trial_id": trial_id_value,
                    "label": label,
                    "confidence": confidence,
                    "actual_model": actual_model,
                    **validated_metadata,
                }
            )
            successful_rows_validated += 1
        normalised.append(item)

    models = list(expected_models)

    successes: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    seen_trial_ids: dict[str, dict[str, Any]] = {}
    duplicate_file_copies = 0
    failures_by_slot: Counter[tuple[str, str, str, int]] = Counter()
    for row in normalised:
        key = (
            row["model_key"],
            row["condition"],
            row["image_id"],
            row["run_number"],
        )
        if row["status"] != "success":
            failures_by_slot[key] += 1
            continue
        trial_id = row["trial_id"]
        if trial_id and trial_id in seen_trial_ids:
            existing = seen_trial_ids[trial_id]
            if successful_trial_signature(existing) != successful_trial_signature(row):
                raise AnalysisInputError(
                    f"Conflicting successful rows share trial_id {trial_id}: "
                    f"{existing['source']}:{existing['source_row']} and "
                    f"{row['source']}:{row['source_row']}. Duplicate rows may only "
                    "be ignored when slot, label, confidence, protocol/prompt/image/"
                    "model metadata, and actual_model are identical."
                )
            duplicate_file_copies += 1
            continue
        if key in successes:
            existing = successes[key]
            raise AnalysisInputError(
                "Multiple successful rows exist for one trial slot: "
                f"{key}; rows {existing['source']}:{existing['source_row']} and "
                f"{row['source']}:{row['source_row']}."
            )
        successes[key] = row
        if trial_id:
            seen_trial_ids[trial_id] = row

    actual_model_by_model: dict[str, str | None] = {}
    for model in models:
        model_rows = [row for row in successes.values() if row["model_key"] == model]
        actual_models = {row["actual_model"] for row in model_rows}
        if len(actual_models) > 1:
            raise AnalysisInputError(
                f"Successful rows for model {model} contain multiple actual_model "
                f"values: {sorted(actual_models)}. Analyse model revisions separately."
            )
        actual_model_by_model[model] = next(iter(actual_models), None)

    expected_slots = [
        (model, condition, image_id, run_number)
        for model in models
        for condition in CONDITIONS
        for image_id in expected_images
        for run_number in range(1, repeats + 1)
    ]
    missing_slots = [key for key in expected_slots if key not in successes]
    completeness = {
        "expected_successful_trials": len(expected_slots),
        "observed_successful_trials": len(expected_slots) - len(missing_slots),
        "missing_successful_trials": len(missing_slots),
        "complete": not missing_slots,
        "failure_rows_in_scope": sum(failures_by_slot.values()),
        "duplicate_file_copies_ignored": duplicate_file_copies,
        "ignored_out_of_scope_rows": ignored_scope_rows,
        "successful_rows_with_frozen_metadata_validated": successful_rows_validated,
        "actual_model_by_model": actual_model_by_model,
        "missing_slots": [
            {
                "model_key": model,
                "condition": condition,
                "image_id": image_id,
                "run_number": run_number,
                "failure_rows_for_slot": failures_by_slot[(model, condition, image_id, run_number)],
            }
            for model, condition, image_id, run_number in missing_slots
        ],
    }
    return {
        "models": models,
        "successes": successes,
        "actual_model_by_model": actual_model_by_model,
        "completeness": completeness,
    }


def aggregate_image_conditions(
    *,
    models: Sequence[str],
    evaluation_images: Sequence[str],
    repeats: int,
    successes: dict[tuple[str, str, str, int], dict[str, Any]],
    human: dict[str, Any],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    base_rates = human["base_rates"]
    sensitivity_rates = human["sensitivity_rates"]
    for model in models:
        for condition in CONDITIONS:
            for image_id in evaluation_images:
                trials = [
                    successes[(model, condition, image_id, run_number)]
                    for run_number in range(1, repeats + 1)
                    if (model, condition, image_id, run_number) in successes
                ]
                labels = [trial["label"] for trial in trials]
                confidences = [trial["confidence"] for trial in trials]
                if labels:
                    mode, mode_count, tied_modes = modal_label(labels)
                    expected_support = statistics.fmean(
                        base_rates[(image_id, label)] for label in labels
                    )
                    sensitivity_expected = (
                        statistics.fmean(
                            sensitivity_rates[(image_id, label)] for label in labels
                        )
                        if sensitivity_rates
                        else None
                    )
                    sensitivity_plurality = human["sensitivity_plurality"]
                    row = {
                        "model_key": model,
                        "condition": condition,
                        "image_id": image_id,
                        "human_n_responses": human["n_by_image"][image_id],
                        "n_expected_repeats": repeats,
                        "n_successful_repeats": len(trials),
                        "complete": len(trials) == repeats,
                        "repeat_labels": labels,
                        "label_counts": {
                            label: labels.count(label)
                            for label in VALID_LABELS
                            if label in labels
                        },
                        "modal_label": mode,
                        "modal_count": mode_count,
                        "modal_tied_labels": tied_modes,
                        "expected_human_endorsement": expected_support,
                        "modal_label_human_support": base_rates[(image_id, mode)],
                        "modal_label_plurality_match": mode in human["plurality"][image_id],
                        "stability": mode_count / len(labels),
                        "pairwise_label_agreement": pairwise_label_agreement(labels),
                        "mean_confidence": statistics.fmean(confidences),
                        "mixed_none_removed_expected_endorsement": sensitivity_expected,
                        "mixed_none_removed_modal_support": (
                            sensitivity_rates[(image_id, mode)]
                            if sensitivity_rates
                            else None
                        ),
                        "mixed_none_removed_plurality_match": (
                            mode in sensitivity_plurality[image_id]
                            if sensitivity_plurality
                            else None
                        ),
                        "actual_models": sorted(
                            {
                                trial["actual_model"]
                                for trial in trials
                                if trial["actual_model"]
                            }
                        ),
                    }
                else:
                    row = {
                        "model_key": model,
                        "condition": condition,
                        "image_id": image_id,
                        "human_n_responses": human["n_by_image"][image_id],
                        "n_expected_repeats": repeats,
                        "n_successful_repeats": 0,
                        "complete": False,
                        "repeat_labels": [],
                        "label_counts": {},
                        "modal_label": None,
                        "modal_count": 0,
                        "modal_tied_labels": [],
                        "expected_human_endorsement": None,
                        "modal_label_human_support": None,
                        "modal_label_plurality_match": None,
                        "stability": None,
                        "pairwise_label_agreement": None,
                        "mean_confidence": None,
                        "mixed_none_removed_expected_endorsement": None,
                        "mixed_none_removed_modal_support": None,
                        "mixed_none_removed_plurality_match": None,
                        "actual_models": [],
                    }
                summaries.append(row)
    return summaries


def condition_summaries(
    image_rows: Sequence[dict[str, Any]], models: Sequence[str]
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model in models:
        for condition in CONDITIONS:
            rows = [
                row
                for row in image_rows
                if row["model_key"] == model and row["condition"] == condition
            ]
            available = [row for row in rows if row["n_successful_repeats"] > 0]
            summaries.append(
                {
                    "model_key": model,
                    "condition": condition,
                    "n_expected_images": len(rows),
                    "n_images_with_success": len(available),
                    "n_complete_images": sum(row["complete"] for row in rows),
                    "complete": bool(rows) and all(row["complete"] for row in rows),
                    "macro_expected_human_endorsement": mean_or_none(
                        row["expected_human_endorsement"] for row in available
                    ),
                    "micro_expected_human_endorsement": weighted_mean_or_none(
                        (
                            row["expected_human_endorsement"],
                            row["human_n_responses"],
                        )
                        for row in available
                    ),
                    "macro_modal_label_human_support": mean_or_none(
                        row["modal_label_human_support"] for row in available
                    ),
                    "plurality_matches": sum(
                        row["modal_label_plurality_match"] is True for row in available
                    ),
                    "plurality_match_rate": mean_or_none(
                        float(row["modal_label_plurality_match"])
                        for row in available
                        if row["modal_label_plurality_match"] is not None
                    ),
                    "mean_stability": mean_or_none(row["stability"] for row in available),
                    "mean_pairwise_label_agreement": mean_or_none(
                        row["pairwise_label_agreement"] for row in available
                    ),
                    "mean_confidence": mean_or_none(
                        row["mean_confidence"] for row in available
                    ),
                    "mixed_none_removed_macro_expected_endorsement": mean_or_none(
                        row["mixed_none_removed_expected_endorsement"]
                        for row in available
                    ),
                    "mixed_none_removed_micro_expected_endorsement": weighted_mean_or_none(
                        (
                            row["mixed_none_removed_expected_endorsement"],
                            row["human_n_responses"],
                        )
                        for row in available
                    ),
                    "mixed_none_removed_macro_modal_support": mean_or_none(
                        row["mixed_none_removed_modal_support"] for row in available
                    ),
                    "mixed_none_removed_plurality_matches": sum(
                        row["mixed_none_removed_plurality_match"] is True
                        for row in available
                    ),
                    "mixed_none_removed_plurality_match_rate": mean_or_none(
                        float(row["mixed_none_removed_plurality_match"])
                        for row in available
                        if row["mixed_none_removed_plurality_match"] is not None
                    ),
                    "actual_models": sorted(
                        {
                            actual_model
                            for row in available
                            for actual_model in row["actual_models"]
                        }
                    ),
                }
            )
    return summaries


def delta_direction(delta: float) -> str:
    if delta > EPSILON:
        return "improved"
    if delta < -EPSILON:
        return "regressed"
    return "unchanged"


def plurality_transition(before: bool, after: bool) -> str:
    if not before and after:
        return "wrong_to_right"
    if before and not after:
        return "right_to_wrong"
    if before and after:
        return "matched_both"
    return "matched_neither"


def paired_metric_statistics(
    before: Sequence[float],
    after: Sequence[float],
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if len(before) != len(after):
        raise ValueError("Paired metric vectors must have the same length.")
    deltas = [right - left for left, right in zip(before, after)]
    ci_low, ci_high = bootstrap_mean_ci(
        deltas, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    return {
        "n_paired_images": len(deltas),
        "mean_before": mean_or_none(before),
        "mean_after": mean_or_none(after),
        "mean_delta": mean_or_none(deltas),
        "median_delta": statistics.median(deltas) if deltas else None,
        "improved_images": sum(delta > EPSILON for delta in deltas),
        "regressed_images": sum(delta < -EPSILON for delta in deltas),
        "unchanged_images": sum(abs(delta) <= EPSILON for delta in deltas),
        "exact_sign_flip_p_two_sided": exact_sign_flip_p(deltas),
        "bootstrap_percentile_ci_95": [ci_low, ci_high],
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
    }


def build_paired_comparisons(
    *,
    image_rows: Sequence[dict[str, Any]],
    models: Sequence[str],
    evaluation_images: Sequence[str],
    config: dict[str, Any],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {
        (row["model_key"], row["condition"], row["image_id"]): row
        for row in image_rows
    }
    comparison_summaries: list[dict[str, Any]] = []
    image_deltas: list[dict[str, Any]] = []
    configured_primary = tuple(config.get("primary_comparison", ("p0", "p2")))
    primary_model = config.get("primary_model_key")

    for model in models:
        for comparison_id, condition_before, condition_after in COMPARISONS:
            pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
            pair_image_ids: list[str] = []
            for image_id in evaluation_images:
                before = index[(model, condition_before, image_id)]
                after = index[(model, condition_after, image_id)]
                if (
                    before["expected_human_endorsement"] is not None
                    and after["expected_human_endorsement"] is not None
                ):
                    pairs.append((before, after))
                    pair_image_ids.append(image_id)

            expected_before = [pair[0]["expected_human_endorsement"] for pair in pairs]
            expected_after = [pair[1]["expected_human_endorsement"] for pair in pairs]
            modal_before = [pair[0]["modal_label_human_support"] for pair in pairs]
            modal_after = [pair[1]["modal_label_human_support"] for pair in pairs]
            stability_before = [pair[0]["stability"] for pair in pairs]
            stability_after = [pair[1]["stability"] for pair in pairs]
            confidence_before = [pair[0]["mean_confidence"] for pair in pairs]
            confidence_after = [pair[1]["mean_confidence"] for pair in pairs]
            match_before = [bool(pair[0]["modal_label_plurality_match"]) for pair in pairs]
            match_after = [bool(pair[1]["modal_label_plurality_match"]) for pair in pairs]
            transition_counts = Counter(
                plurality_transition(left, right)
                for left, right in zip(match_before, match_after)
            )
            wrong_to_right = transition_counts["wrong_to_right"]
            right_to_wrong = transition_counts["right_to_wrong"]

            expected_stats = paired_metric_statistics(
                expected_before,
                expected_after,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            )
            modal_stats = paired_metric_statistics(
                modal_before,
                modal_after,
                bootstrap_seed=bootstrap_seed,
                bootstrap_replicates=bootstrap_replicates,
            )

            sensitivity_available = bool(pairs) and all(
                pair[0]["mixed_none_removed_expected_endorsement"] is not None
                and pair[1]["mixed_none_removed_expected_endorsement"] is not None
                for pair in pairs
            )
            sensitivity: dict[str, Any] = {"available": sensitivity_available}
            if sensitivity_available:
                sensitivity_expected_before = [
                    pair[0]["mixed_none_removed_expected_endorsement"] for pair in pairs
                ]
                sensitivity_expected_after = [
                    pair[1]["mixed_none_removed_expected_endorsement"] for pair in pairs
                ]
                sensitivity_modal_before = [
                    pair[0]["mixed_none_removed_modal_support"] for pair in pairs
                ]
                sensitivity_modal_after = [
                    pair[1]["mixed_none_removed_modal_support"] for pair in pairs
                ]
                sensitivity_match_before = [
                    bool(pair[0]["mixed_none_removed_plurality_match"])
                    for pair in pairs
                ]
                sensitivity_match_after = [
                    bool(pair[1]["mixed_none_removed_plurality_match"])
                    for pair in pairs
                ]
                sensitivity_transitions = Counter(
                    plurality_transition(left, right)
                    for left, right in zip(
                        sensitivity_match_before, sensitivity_match_after
                    )
                )
                sensitivity.update(
                    {
                        "expected_human_endorsement": paired_metric_statistics(
                            sensitivity_expected_before,
                            sensitivity_expected_after,
                            bootstrap_seed=bootstrap_seed,
                            bootstrap_replicates=bootstrap_replicates,
                        ),
                        "modal_label_human_support": paired_metric_statistics(
                            sensitivity_modal_before,
                            sensitivity_modal_after,
                            bootstrap_seed=bootstrap_seed,
                            bootstrap_replicates=bootstrap_replicates,
                        ),
                        "plurality": {
                            "matches_before": sum(sensitivity_match_before),
                            "matches_after": sum(sensitivity_match_after),
                            "wrong_to_right": sensitivity_transitions["wrong_to_right"],
                            "right_to_wrong": sensitivity_transitions["right_to_wrong"],
                            "matched_both": sensitivity_transitions["matched_both"],
                            "matched_neither": sensitivity_transitions["matched_neither"],
                            "exact_mcnemar_p_two_sided": exact_mcnemar_p(
                                sensitivity_transitions["wrong_to_right"],
                                sensitivity_transitions["right_to_wrong"],
                            ),
                        },
                    }
                )

            label_transition_counts = Counter(
                (pair[0]["modal_label"], pair[1]["modal_label"]) for pair in pairs
            )
            label_order = {label: position for position, label in enumerate(VALID_LABELS)}
            label_transitions = [
                {"from": before_label, "to": after_label, "count": count}
                for (before_label, after_label), count in sorted(
                    label_transition_counts.items(),
                    key=lambda item: (
                        label_order[item[0][0]], label_order[item[0][1]]
                    ),
                )
            ]
            is_primary = (
                condition_before,
                condition_after,
            ) == configured_primary and model == primary_model
            summary = {
                "model_key": model,
                "comparison": comparison_id,
                "condition_before": condition_before,
                "condition_after": condition_after,
                "is_primary_model_and_comparison": is_primary,
                "n_expected_paired_images": len(evaluation_images),
                "n_paired_images": len(pairs),
                "n_complete_paired_images": sum(
                    before["complete"] and after["complete"]
                    for before, after in pairs
                ),
                "complete": len(pairs) == len(evaluation_images)
                and all(before["complete"] and after["complete"] for before, after in pairs),
                "expected_human_endorsement": expected_stats,
                "modal_label_human_support": modal_stats,
                "plurality": {
                    "matches_before": sum(match_before),
                    "matches_after": sum(match_after),
                    "match_rate_before": mean_or_none(float(value) for value in match_before),
                    "match_rate_after": mean_or_none(float(value) for value in match_after),
                    "wrong_to_right": wrong_to_right,
                    "right_to_wrong": right_to_wrong,
                    "matched_both": transition_counts["matched_both"],
                    "matched_neither": transition_counts["matched_neither"],
                    "exact_mcnemar_p_two_sided": exact_mcnemar_p(
                        wrong_to_right, right_to_wrong
                    ),
                },
                "stability": {
                    "mean_before": mean_or_none(stability_before),
                    "mean_after": mean_or_none(stability_after),
                    "mean_delta": mean_or_none(
                        right - left
                        for left, right in zip(stability_before, stability_after)
                    ),
                },
                "mean_confidence": {
                    "mean_before": mean_or_none(confidence_before),
                    "mean_after": mean_or_none(confidence_after),
                    "mean_delta": mean_or_none(
                        right - left
                        for left, right in zip(confidence_before, confidence_after)
                    ),
                },
                "improved_images": [
                    image_id
                    for image_id, left, right in zip(
                        pair_image_ids, expected_before, expected_after
                    )
                    if right - left > EPSILON
                ],
                "regressed_images": [
                    image_id
                    for image_id, left, right in zip(
                        pair_image_ids, expected_before, expected_after
                    )
                    if right - left < -EPSILON
                ],
                "unchanged_images": [
                    image_id
                    for image_id, left, right in zip(
                        pair_image_ids, expected_before, expected_after
                    )
                    if abs(right - left) <= EPSILON
                ],
                "modal_label_transitions": label_transitions,
                "mixed_none_removed_sensitivity": sensitivity,
            }
            comparison_summaries.append(summary)

            for image_id, (before, after) in zip(pair_image_ids, pairs):
                expected_delta = (
                    after["expected_human_endorsement"]
                    - before["expected_human_endorsement"]
                )
                sensitivity_delta = (
                    after["mixed_none_removed_expected_endorsement"]
                    - before["mixed_none_removed_expected_endorsement"]
                    if sensitivity_available
                    else None
                )
                image_deltas.append(
                    {
                        "model_key": model,
                        "comparison": comparison_id,
                        "condition_before": condition_before,
                        "condition_after": condition_after,
                        "image_id": image_id,
                        "both_cells_complete": before["complete"] and after["complete"],
                        "expected_endorsement_before": before[
                            "expected_human_endorsement"
                        ],
                        "expected_endorsement_after": after[
                            "expected_human_endorsement"
                        ],
                        "expected_endorsement_delta": expected_delta,
                        "expected_endorsement_direction": delta_direction(expected_delta),
                        "mixed_none_removed_expected_before": before[
                            "mixed_none_removed_expected_endorsement"
                        ],
                        "mixed_none_removed_expected_after": after[
                            "mixed_none_removed_expected_endorsement"
                        ],
                        "mixed_none_removed_expected_delta": sensitivity_delta,
                        "modal_label_before": before["modal_label"],
                        "modal_label_after": after["modal_label"],
                        "modal_label_transition": (
                            f"{before['modal_label']} -> {after['modal_label']}"
                        ),
                        "modal_support_before": before["modal_label_human_support"],
                        "modal_support_after": after["modal_label_human_support"],
                        "modal_support_delta": after["modal_label_human_support"]
                        - before["modal_label_human_support"],
                        "plurality_match_before": before[
                            "modal_label_plurality_match"
                        ],
                        "plurality_match_after": after[
                            "modal_label_plurality_match"
                        ],
                        "plurality_transition": plurality_transition(
                            bool(before["modal_label_plurality_match"]),
                            bool(after["modal_label_plurality_match"]),
                        ),
                        "stability_before": before["stability"],
                        "stability_after": after["stability"],
                        "stability_delta": after["stability"] - before["stability"],
                        "mean_confidence_before": before["mean_confidence"],
                        "mean_confidence_after": after["mean_confidence"],
                        "mean_confidence_delta": after["mean_confidence"]
                        - before["mean_confidence"],
                    }
                )
    return comparison_summaries, image_deltas


def serialisable_image_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        for key, value in list(item.items()):
            if isinstance(value, float):
                item[key] = round_or_none(value)
        result.append(item)
    return result


def round_nested(value: Any) -> Any:
    if isinstance(value, float):
        return round_or_none(value)
    if isinstance(value, list):
        return [round_nested(item) for item in value]
    if isinstance(value, dict):
        return {key: round_nested(item) for key, item in value.items()}
    return value


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=isinstance(value, dict))
    if isinstance(value, float):
        return round(value, 9)
    return value


def atomic_write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: csv_value(row.get(field)) for field in fields})
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def flatten_condition_summary(row: dict[str, Any]) -> dict[str, Any]:
    return dict(row)


def flatten_comparison_summary(row: dict[str, Any]) -> dict[str, Any]:
    expected = row["expected_human_endorsement"]
    modal = row["modal_label_human_support"]
    plurality = row["plurality"]
    stability = row["stability"]
    confidence = row["mean_confidence"]
    sensitivity = row["mixed_none_removed_sensitivity"]
    sensitivity_expected = sensitivity.get("expected_human_endorsement", {})
    sensitivity_plurality = sensitivity.get("plurality", {})
    ci = expected["bootstrap_percentile_ci_95"]
    sensitivity_ci = sensitivity_expected.get("bootstrap_percentile_ci_95", [None, None])
    return {
        "model_key": row["model_key"],
        "comparison": row["comparison"],
        "condition_before": row["condition_before"],
        "condition_after": row["condition_after"],
        "is_primary_model_and_comparison": row[
            "is_primary_model_and_comparison"
        ],
        "complete": row["complete"],
        "n_expected_paired_images": row["n_expected_paired_images"],
        "n_paired_images": row["n_paired_images"],
        "n_complete_paired_images": row["n_complete_paired_images"],
        "expected_endorsement_before": expected["mean_before"],
        "expected_endorsement_after": expected["mean_after"],
        "expected_endorsement_delta": expected["mean_delta"],
        "expected_endorsement_ci_low": ci[0],
        "expected_endorsement_ci_high": ci[1],
        "expected_endorsement_sign_flip_p": expected[
            "exact_sign_flip_p_two_sided"
        ],
        "improved_images": expected["improved_images"],
        "regressed_images": expected["regressed_images"],
        "unchanged_images": expected["unchanged_images"],
        "modal_support_before": modal["mean_before"],
        "modal_support_after": modal["mean_after"],
        "modal_support_delta": modal["mean_delta"],
        "plurality_matches_before": plurality["matches_before"],
        "plurality_matches_after": plurality["matches_after"],
        "wrong_to_right": plurality["wrong_to_right"],
        "right_to_wrong": plurality["right_to_wrong"],
        "mcnemar_exact_p": plurality["exact_mcnemar_p_two_sided"],
        "stability_before": stability["mean_before"],
        "stability_after": stability["mean_after"],
        "stability_delta": stability["mean_delta"],
        "confidence_before": confidence["mean_before"],
        "confidence_after": confidence["mean_after"],
        "confidence_delta": confidence["mean_delta"],
        "mixed_none_sensitivity_available": sensitivity["available"],
        "mixed_none_expected_delta": sensitivity_expected.get("mean_delta"),
        "mixed_none_ci_low": sensitivity_ci[0],
        "mixed_none_ci_high": sensitivity_ci[1],
        "mixed_none_sign_flip_p": sensitivity_expected.get(
            "exact_sign_flip_p_two_sided"
        ),
        "mixed_none_wrong_to_right": sensitivity_plurality.get("wrong_to_right"),
        "mixed_none_right_to_wrong": sensitivity_plurality.get("right_to_wrong"),
        "mixed_none_mcnemar_exact_p": sensitivity_plurality.get(
            "exact_mcnemar_p_two_sided"
        ),
    }


def format_percent(value: float | None) -> str:
    return "NA" if value is None else f"{100 * value:.1f}%"


def format_number(value: float | None, digits: int = 4) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def render_results_markdown(
    *,
    analysis_status: str,
    condition_rows: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    completeness: dict[str, Any],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> str:
    lines = [
        "# Prompt engineering experiment results",
        "",
        f"Analysis status: **{analysis_status.upper()}**.",
        "",
        "Repeated calls were collapsed within each model, condition, and image. "
        "All intervals and tests use images as the inferential units; repeats are "
        "not counted as independent observations.",
        "",
    ]
    if analysis_status != "complete":
        lines.extend(
            [
                "> Incomplete analysis: some planned successful trials are missing. "
                "Inferential results below use only available paired images and must "
                "not be reported as the final experiment.",
                "",
            ]
        )
    lines.extend(
        [
            "## Completeness",
            "",
            f"- Expected successful trials: {completeness['expected_successful_trials']}",
            f"- Observed successful trials: {completeness['observed_successful_trials']}",
            f"- Missing successful trials: {completeness['missing_successful_trials']}",
            f"- Failure rows retained in trial logs: {completeness['failure_rows_in_scope']}",
            "",
            "## Condition summaries",
            "",
            "| Model | Condition | Complete images | Expected endorsement "
            "(image macro) | Expected endorsement (response micro) | Modal-label "
            "support | Plurality matches | Stability | Mean confidence |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in condition_rows:
        lines.append(
            f"| {row['model_key']} | {row['condition'].upper()} | "
            f"{row['n_complete_images']}/{row['n_expected_images']} | "
            f"{format_percent(row['macro_expected_human_endorsement'])} | "
            f"{format_percent(row['micro_expected_human_endorsement'])} | "
            f"{format_percent(row['macro_modal_label_human_support'])} | "
            f"{row['plurality_matches']}/{row['n_images_with_success']} | "
            f"{format_number(row['mean_stability'], 3)} | "
            f"{format_number(row['mean_confidence'], 1)} |"
        )

    lines.extend(
        [
            "",
            "Expected human endorsement is the mean participant endorsement of "
            "the labels selected across repeats, calculated within each image and "
            "then macro-averaged across images for the primary estimand. The response-"
            "micro column is a pre-specified secondary summary weighted by each "
            "image's human n_responses. Stability is the modal label's share of "
            "successful repeats.",
            "",
            "## Paired comparisons",
            "",
            "| Model | Comparison | Paired images | Mean delta | 95% image-bootstrap CI | "
            "Exact sign-flip p | Improved / regressed | Plurality W→R / R→W | "
            "Exact McNemar p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        expected = row["expected_human_endorsement"]
        plurality = row["plurality"]
        ci_low, ci_high = expected["bootstrap_percentile_ci_95"]
        primary_marker = " (primary)" if row["is_primary_model_and_comparison"] else ""
        lines.append(
            f"| {row['model_key']} | {row['comparison']}{primary_marker} | "
            f"{row['n_paired_images']}/{row['n_expected_paired_images']} | "
            f"{format_percent(expected['mean_delta'])} | "
            f"[{format_percent(ci_low)}, {format_percent(ci_high)}] | "
            f"{format_number(expected['exact_sign_flip_p_two_sided'])} | "
            f"{expected['improved_images']} / {expected['regressed_images']} | "
            f"{plurality['wrong_to_right']} / {plurality['right_to_wrong']} | "
            f"{format_number(plurality['exact_mcnemar_p_two_sided'])} |"
        )

    lines.extend(
        [
            "",
            "The two-sided sign-flip test exactly enumerates all image-level sign "
            "assignments. The percentile confidence interval resamples images with "
            f"replacement ({bootstrap_replicates:,} draws; seed {bootstrap_seed}). "
            "McNemar's exact test uses only discordant modal-label plurality matches.",
            "",
            "## Mixed-None sensitivity",
            "",
            "The sensitivity analysis removes the three None endorsements that "
            "were co-selected with a moral category, as an alternative interpretation "
            "of those mixed responses. It uses "
            "the same image-level aggregation and paired tests.",
            "",
            "### Condition summaries",
            "",
            "| Model | Condition | Expected endorsement (image macro) | "
            "Expected endorsement (response micro) |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in condition_rows:
        lines.append(
            f"| {row['model_key']} | {row['condition'].upper()} | "
            f"{format_percent(row['mixed_none_removed_macro_expected_endorsement'])} | "
            f"{format_percent(row['mixed_none_removed_micro_expected_endorsement'])} |"
        )

    lines.extend(
        [
            "",
            "### Paired comparisons",
            "",
            "| Model | Comparison | Mean delta | 95% image-bootstrap CI | "
            "Exact sign-flip p |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in comparisons:
        sensitivity = row["mixed_none_removed_sensitivity"]
        expected = sensitivity.get("expected_human_endorsement")
        if not sensitivity["available"] or not expected:
            lines.append(f"| {row['model_key']} | {row['comparison']} | NA | NA | NA |")
            continue
        ci_low, ci_high = expected["bootstrap_percentile_ci_95"]
        lines.append(
            f"| {row['model_key']} | {row['comparison']} | "
            f"{format_percent(expected['mean_delta'])} | "
            f"[{format_percent(ci_low)}, {format_percent(ci_high)}] | "
            f"{format_number(expected['exact_sign_flip_p_two_sided'])} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- This is an exploratory 16-image pilot, not an out-of-sample accuracy benchmark.",
            "- Participant responses were multi-label endorsements; within-image rates need not sum to 100%.",
            "- Confidence is model-reported and is not a calibrated probability.",
            "- Repeated calls measure output stability but do not increase the inferential sample size above the number of images.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    payload: dict[str, Any],
    image_rows: Sequence[dict[str, Any]],
    condition_rows: Sequence[dict[str, Any]],
    comparisons: Sequence[dict[str, Any]],
    image_deltas: Sequence[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        output_dir / "analysis_results.json",
        json.dumps(round_nested(payload), ensure_ascii=False, indent=2) + "\n",
    )

    image_fields = [
        "model_key",
        "condition",
        "image_id",
        "human_n_responses",
        "n_expected_repeats",
        "n_successful_repeats",
        "complete",
        "repeat_labels",
        "label_counts",
        "modal_label",
        "modal_count",
        "modal_tied_labels",
        "expected_human_endorsement",
        "modal_label_human_support",
        "modal_label_plurality_match",
        "stability",
        "pairwise_label_agreement",
        "mean_confidence",
        "mixed_none_removed_expected_endorsement",
        "mixed_none_removed_modal_support",
        "mixed_none_removed_plurality_match",
        "actual_models",
    ]
    atomic_write_csv(output_dir / "image_condition_metrics.csv", image_rows, image_fields)

    condition_fields = list(condition_rows[0]) if condition_rows else []
    atomic_write_csv(
        output_dir / "condition_summary.csv", condition_rows, condition_fields
    )

    flat_comparisons = [flatten_comparison_summary(row) for row in comparisons]
    comparison_fields = list(flat_comparisons[0]) if flat_comparisons else []
    atomic_write_csv(
        output_dir / "paired_comparisons.csv",
        flat_comparisons,
        comparison_fields,
    )

    delta_fields = list(image_deltas[0]) if image_deltas else []
    atomic_write_csv(output_dir / "image_pair_deltas.csv", image_deltas, delta_fields)

    atomic_write_text(
        output_dir / "RESULTS.md",
        render_results_markdown(
            analysis_status=payload["analysis_status"],
            condition_rows=condition_rows,
            comparisons=comparisons,
            completeness=payload["completeness"],
            bootstrap_seed=payload["methods"]["bootstrap_seed"],
            bootstrap_replicates=payload["methods"]["bootstrap_replicates"],
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate repeat trials by image and compare P0/P1/P2 against "
            "anonymous human-endorsement targets."
        )
    )
    parser.add_argument(
        "--human-targets",
        type=Path,
        default=EXPERIMENT_DIR / "data" / "human_endorsements_16.csv",
    )
    parser.add_argument(
        "--results-root", type=Path, default=EXPERIMENT_DIR / "results"
    )
    parser.add_argument(
        "--trials-csv",
        type=Path,
        action="append",
        default=[],
        help="Explicit trial CSV; repeat for multiple files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EXPERIMENT_DIR / "results" / "analysis",
    )
    parser.add_argument(
        "--config", type=Path, default=EXPERIMENT_DIR / "experiment_config.json"
    )
    parser.add_argument(
        "--evaluation-manifest",
        type=Path,
        default=EXPERIMENT_DIR / "manifests" / "evaluation_images_16.csv",
    )
    parser.add_argument(
        "--protocol-manifest",
        type=Path,
        default=EXPERIMENT_DIR / "protocol_manifest.json",
    )
    parser.add_argument(
        "--schedule",
        "--schedule-manifest",
        dest="schedule_manifest",
        type=Path,
        default=EXPERIMENT_DIR / "manifests" / "schedule.csv",
        help="Frozen schedule.csv used by the runner.",
    )
    parser.add_argument("--protocol-sha256")
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Configured model keys to analyse. If omitted, every model in "
            "experiment_config.json model_specs is required."
        ),
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write clearly marked partial outputs instead of failing on missing trials.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        help=(
            "Override experiment_config.json primary_inference.bootstrap_seed. "
            "Normally omit this to follow the frozen protocol."
        ),
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        help=(
            "Override experiment_config.json primary_inference.bootstrap_samples. "
            "Normally omit this to follow the frozen protocol."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    primary_inference = config.get("primary_inference", {})
    if primary_inference and not isinstance(primary_inference, dict):
        raise AnalysisInputError("primary_inference must be a JSON object.")
    bootstrap_seed = (
        args.bootstrap_seed
        if args.bootstrap_seed is not None
        else int(primary_inference.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    )
    bootstrap_replicates = (
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else int(
            primary_inference.get(
                "bootstrap_samples", DEFAULT_BOOTSTRAP_REPLICATES
            )
        )
    )
    alpha = float(primary_inference.get("alpha", 0.05))
    if bootstrap_replicates < 1:
        raise AnalysisInputError("Configured bootstrap_samples must be positive.")
    if not 0 < alpha < 1:
        raise AnalysisInputError("Configured primary-inference alpha must be in (0, 1).")
    if config.get("inference_unit", "image") != "image":
        raise AnalysisInputError("The frozen analysis requires inference_unit='image'.")
    repeats = int(config.get("repeats", 3))
    if repeats != 3:
        raise AnalysisInputError(
            f"This frozen analysis expects three repeats; config specifies {repeats}."
        )
    evaluation_images = load_evaluation_images(args.evaluation_manifest)
    if len(evaluation_images) != 16:
        raise AnalysisInputError(
            f"Expected 16 evaluation images; found {len(evaluation_images)}."
        )
    human = load_human_targets(args.human_targets, evaluation_images)

    if not args.protocol_manifest.is_file():
        raise FileNotFoundError(
            f"Frozen protocol manifest not found: {args.protocol_manifest}"
        )
    protocol_manifest = read_json(args.protocol_manifest)
    schedule = load_frozen_schedule(
        args.schedule_manifest,
        expected_images=evaluation_images,
        repeats=repeats,
    )
    frozen = validate_frozen_protocol(
        manifest=protocol_manifest,
        config=config,
        config_path=args.config,
        evaluation_manifest_path=args.evaluation_manifest,
        schedule_path=args.schedule_manifest,
        expected_images=evaluation_images,
    )
    expected_models = resolve_models(config, args.models)

    trial_files = discover_trial_files(args.results_root, args.trials_csv)
    raw_trials, trial_sources = load_trial_rows(trial_files)
    selected_protocol_hash = choose_protocol_hash(
        args.protocol_sha256, protocol_manifest
    )
    trials = normalise_trials(
        raw_trials,
        selected_protocol_hash=selected_protocol_hash,
        expected_images=evaluation_images,
        repeats=repeats,
        expected_models=expected_models,
        frozen=frozen,
        schedule=schedule,
    )
    completeness = trials["completeness"]
    if not completeness["complete"] and not args.allow_incomplete:
        preview = ", ".join(
            f"{row['model_key']}/{row['condition']}/{row['image_id']}/run{row['run_number']}"
            for row in completeness["missing_slots"][:12]
        )
        raise IncompleteExperimentError(
            "INCOMPLETE: missing "
            f"{completeness['missing_successful_trials']} of "
            f"{completeness['expected_successful_trials']} required successful "
            f"trials. First missing slots: {preview}. Rerun inference, or use "
            "--allow-incomplete for explicitly provisional outputs."
        )

    image_rows = aggregate_image_conditions(
        models=trials["models"],
        evaluation_images=evaluation_images,
        repeats=repeats,
        successes=trials["successes"],
        human=human,
    )
    condition_rows = condition_summaries(image_rows, trials["models"])
    comparisons, image_deltas = build_paired_comparisons(
        image_rows=image_rows,
        models=trials["models"],
        evaluation_images=evaluation_images,
        config=config,
        bootstrap_seed=bootstrap_seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    analysis_status = "complete" if completeness["complete"] else "incomplete"
    payload = {
        "schema_version": "prompt-experiment-analysis-v1",
        "generated_at_utc": utc_now(),
        "analysis_status": analysis_status,
        "protocol_id": frozen["protocol_id"],
        "protocol_sha256": selected_protocol_hash,
        "primary_model_key": config.get("primary_model_key"),
        "primary_comparison": config.get("primary_comparison", ["p0", "p2"]),
        "models": trials["models"],
        "actual_model_by_model": trials["actual_model_by_model"],
        "conditions": list(CONDITIONS),
        "evaluation_images": evaluation_images,
        "inputs": {
            "analysis_script": {
                "filename": Path(__file__).name,
                "sha256": sha256_file(Path(__file__).resolve()),
            },
            "human_targets": {
                "filename": human["filename"],
                "sha256": human["sha256"],
                "rows": human["row_count"],
                "contains_participant_pii": False,
            },
            "trial_csvs": trial_sources,
            "protocol_manifest": {
                "filename": args.protocol_manifest.name,
                "sha256": sha256_file(args.protocol_manifest),
                "protocol_sha256": selected_protocol_hash,
            },
            "schedule": {
                "filename": args.schedule_manifest.name,
                "sha256": sha256_file(args.schedule_manifest),
                "slots_per_model": len(schedule),
            },
            "evaluation_manifest": {
                "filename": args.evaluation_manifest.name,
                "sha256": sha256_file(args.evaluation_manifest),
            },
            "config": {
                "filename": args.config.name,
                "sha256": sha256_file(args.config),
            },
        },
        "methods": {
            "inference_unit": "image",
            "repeats_are_independent_samples": False,
            "repeats_per_image_condition": repeats,
            "expected_human_endorsement": (
                "Within-image mean human endorsement of the labels selected "
                "across successful repeats."
            ),
            "condition_macro_expected_endorsement": (
                "Unweighted mean of image-level expected endorsements; this is "
                "the primary aggregation."
            ),
            "condition_micro_expected_endorsement": (
                "Secondary mean of image-level expected endorsements weighted by "
                "each image's human n_responses."
            ),
            "modal_tie_break_order": list(VALID_LABELS),
            "stability": "Modal label count divided by successful repeats.",
            "pairwise_label_agreement": (
                "Proportion of repeat pairs with the same label."
            ),
            "sign_flip_test": (
                "Two-sided exact enumeration of signs for paired image-level deltas."
            ),
            "bootstrap": "Percentile bootstrap resampling paired images with replacement.",
            "bootstrap_seed": bootstrap_seed,
            "bootstrap_replicates": bootstrap_replicates,
            "alpha": alpha,
            "configured_primary_inference": primary_inference,
            "configured_repeat_policy": config.get("repeat_policy"),
            "configured_mixed_none_policy": config.get("mixed_none_policy"),
            "mcnemar": "Two-sided exact binomial McNemar test on discordant images.",
            "mixed_none_sensitivity_available": bool(human["sensitivity_rates"]),
        },
        "completeness": completeness,
        "condition_summaries": condition_rows,
        "paired_comparisons": comparisons,
        "image_condition_metrics": image_rows,
        "image_pair_deltas": image_deltas,
    }
    payload = round_nested(payload)
    image_rows = serialisable_image_rows(image_rows)
    condition_rows = round_nested(condition_rows)
    comparisons = round_nested(comparisons)
    image_deltas = round_nested(image_deltas)
    write_outputs(
        output_dir=args.output_dir,
        payload=payload,
        image_rows=image_rows,
        condition_rows=condition_rows,
        comparisons=comparisons,
        image_deltas=image_deltas,
    )
    print(
        f"Analysis {analysis_status}: {len(trials['models'])} model(s), "
        f"{len(evaluation_images)} images, {repeats} repeats per condition."
    )
    print(f"Wrote analysis outputs to {args.output_dir}")


if __name__ == "__main__":
    try:
        main()
    except (AnalysisInputError, FileNotFoundError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2) from error
