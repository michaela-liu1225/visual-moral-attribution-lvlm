from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import math
import mimetypes
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


SCHEMA_VERSION = "repeatability-v1"
PROMPT_VERSION = "v2-repeatability-2026-09-01"
DEFAULT_EXPERIMENT_ID = "visual-moral-repeatability-v1"

VALID_LABELS = [
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

PROMPT = """
You are evaluating the primary moral concern conveyed by an image.

Base your judgment only on information that is visually present in the
image or can be reasonably inferred from the visible scene. Do not invent
hidden intentions, relationships, events, identities, or background
information that are not supported by the image.

Your task is to select exactly one category that best represents the
PRIMARY moral concern conveyed by the image. The category may reflect
either a visible moral action or a morally relevant condition or situation,
such as vulnerability, suffering, unfair treatment, or violation of dignity.

Use the following categories:

1. Care
Concern for others' well-being, needs, or vulnerability, including
compassion, helping, nurturing, support, and protection from harm.

2. Harm
Physical or emotional suffering, injury, cruelty, abuse, violence, or
actions that cause or contribute to harm to others.

3. Fairness
Concern for justice, equality, rights, impartial treatment, fair
opportunities, or equitable outcomes.

4. Cheating
Violation of fairness through deception, exploitation, rule-breaking,
favoritism, manipulation, or gaining an unfair advantage.

5. Loyalty
Commitment, solidarity, allegiance, or support toward one's group or
its members.

6. Betrayal
Violation of trust or loyalty through disloyalty, abandonment, deception,
or acting against one's group or trusted others.

7. Respect / Authority
Respect for legitimate authority, social roles, rules, hierarchy,
institutions, social order, or established traditions.

8. Subversion
Defiance, rejection, disruption, or undermining of legitimate authority,
established rules, hierarchy, social order, or traditions.

9. Sanctity
Concern for purity, sacredness, dignity, moral integrity, or the
protection of things regarded as morally, culturally, or spiritually
inviolable or worthy of reverence.

10. Degradation
Violation of purity, sacredness, or dignity through contamination,
desecration, humiliation, objectification, or degrading treatment.

11. None of these / Not clearly moral in nature
No moral concern from the categories above is clearly conveyed, or the
visible evidence is insufficient to identify a primary moral concern.

Important classification guidance:

- Classify the PRIMARY moral concern, not every moral theme that could
  possibly be associated with the image.
- Care does not require that helping is already taking place. A clearly
  vulnerable person or someone visibly in need may convey Care when the
  primary moral concern is their welfare or need for support.
- Harm should be preferred when the primary emphasis is visible injury,
  suffering, violence, abuse, cruelty, or the infliction of harm.
- Fairness may include situations involving visibly unequal or unfair
  treatment, even if nobody is actively correcting the unfairness.
- Cheating should be used when there is visible evidence of unfair conduct,
  deception, exploitation, rule-breaking, or unfair advantage, rather than
  merely a general concern about fairness.
- Do not infer discrimination, betrayal, cheating, criminality, harmful
  intent, or other morally relevant background facts solely from a person's
  appearance or identity.
- If several categories are plausible, choose the one that best captures
  the dominant moral meaning of the visible scene.
- Use "None of these / Not clearly moral in nature" when identifying a moral
  category would require substantial unsupported assumptions.

Confidence:
Return an integer from 0 to 100 representing your confidence that the
selected label is the best category for this image.

Use confidence approximately as follows:
- 90-100: very clear visual evidence; little meaningful ambiguity
- 75-89: strong evidence, but some plausible ambiguity
- 55-74: moderate evidence; another category is reasonably plausible
- 30-54: weak or ambiguous evidence
- 0-29: highly uncertain; very little evidence supports a specific category

For the label field, return only one exact category name listed above.

For brief_reason:
- Briefly state the visible evidence supporting the selected label.
- Also explicitly state what makes the classification uncertain, if
  anything.
- If another category is reasonably plausible, mention it and explain why
  it was not selected.
- Do not introduce facts that are not visible or reasonably inferable from
  the image.

Return JSON only with exactly these fields: label, confidence, brief_reason.
Do not wrap the JSON in Markdown code fences.
""".strip()
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


CSV_FIELDS = [
    "schema_version",
    "experiment_id",
    "subset_manifest",
    "subset_manifest_sha256",
    "selection_stratum",
    "run_number",
    "sequence_index",
    "image_id",
    "filename",
    "image_path",
    "provider",
    "requested_model",
    "actual_model",
    "prompt_version",
    "prompt_sha256",
    "temperature_requested",
    "reasoning_mode",
    "provider_reasoning_setting",
    "image_detail",
    "request_started_at_utc",
    "response_received_at_utc",
    "duration_seconds",
    "attempts_used",
    "label",
    "confidence",
    "brief_reason",
    "status",
    "error",
    "response_id",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
    "raw_output",
    "raw_json_path",
]


@dataclass(frozen=True)
class SubsetItem:
    image_id: str
    filename: str
    selection_stratum: str
    image_path: Path


@dataclass(frozen=True)
class ProviderResponse:
    parsed_result: dict[str, Any]
    raw_output: str
    response_id: str
    actual_model: str
    usage: dict[str, int | str]
    response_json: dict[str, Any]


@dataclass(frozen=True)
class ExperimentConfig:
    provider: str
    requested_model: str
    output_csv: Path
    raw_json_dir: Path
    temperature_requested: float | None
    reasoning_mode: str
    provider_reasoning_setting: dict[str, Any]
    image_detail: str


class ProviderRequestError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        raw_output: str = "",
        response_id: str = "",
        actual_model: str = "",
        usage: dict[str, int | str] | None = None,
        response_json: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.response_id = response_id
        self.actual_model = actual_model
        self.usage = usage or {}
        self.response_json = response_json or {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def image_to_data_url(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)
    supported = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if mime_type not in supported:
        raise ValueError(
            f"Unsupported image type for {image_path.name}: {mime_type}"
        )
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def clean_json_output(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```"):].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    return cleaned


def validate_result(result: dict[str, Any]) -> dict[str, Any]:
    label = result.get("label")
    confidence = result.get("confidence")
    brief_reason = result.get("brief_reason")

    if label not in VALID_LABELS:
        raise ValueError(f"Unexpected label returned: {label!r}")
    if isinstance(confidence, bool) or not isinstance(confidence, int):
        raise TypeError("confidence must be an integer")
    if not 0 <= confidence <= 100:
        raise ValueError("confidence must be between 0 and 100")
    if not isinstance(brief_reason, str) or not brief_reason.strip():
        raise ValueError("brief_reason must be a non-empty string")

    return {
        "label": label,
        "confidence": confidence,
        "brief_reason": brief_reason.strip(),
    }


def serialise_sdk_response(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        value = response.model_dump(mode="json")
    elif hasattr(response, "to_dict"):
        value = response.to_dict()
    elif isinstance(response, dict):
        value = response
    else:
        value = {"repr": repr(response)}
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def load_subset_manifest(
    manifest_path: Path,
    image_dir: Path,
) -> tuple[list[SubsetItem], str]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Subset manifest not found: {manifest_path.resolve()}"
        )

    manifest_bytes = manifest_path.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()

    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not 25 <= len(rows) <= 40:
        raise ValueError(
            f"Subset must contain 25-40 images; found {len(rows)}."
        )

    items: list[SubsetItem] = []
    seen_ids: set[str] = set()
    for row in rows:
        image_id = row.get("image_id", "").strip()
        filename = row.get("filename", "").strip()
        stratum = row.get("selection_stratum", "").strip()
        if not image_id or not filename or not stratum:
            raise ValueError(
                "Manifest requires image_id, filename, and selection_stratum."
            )
        if image_id in seen_ids:
            raise ValueError(f"Duplicate subset image ID: {image_id}")
        image_path = image_dir / filename
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Subset image not found: {image_path.resolve()}"
            )
        if image_path.stem != image_id:
            raise ValueError(
                f"Manifest mismatch: {image_id} does not match {filename}."
            )
        seen_ids.add(image_id)
        items.append(
            SubsetItem(
                image_id=image_id,
                filename=filename,
                selection_stratum=stratum,
                image_path=image_path,
            )
        )

    return items, manifest_hash


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output_csv: Path,
    default_raw_json_dir: Path,
) -> None:
    parser.add_argument(
        "--subset-manifest",
        type=Path,
        default=Path("repeatability_subset_32.csv"),
    )
    parser.add_argument("--image-dir", type=Path, default=Path("images"))
    parser.add_argument("--repeats", type=int, default=3, choices=[3, 4, 5])
    parser.add_argument(
        "--run-numbers",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5],
        help=(
            "Run only these repeat numbers. Use this to interleave models "
            "within the same time window."
        ),
    )
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--output-csv", type=Path, default=default_output_csv)
    parser.add_argument(
        "--raw-json-dir",
        type=Path,
        default=default_raw_json_dir,
    )
    parser.add_argument("--request-delay", type=float, default=0.5)
    parser.add_argument("--inter-run-delay", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print the plan without calling an API.",
    )


def _completed_keys(output_csv: Path, experiment_id: str) -> set[tuple[int, str]]:
    if not output_csv.exists():
        return set()
    with output_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != CSV_FIELDS:
            raise ValueError(
                f"Existing CSV has an incompatible header: {output_csv}"
            )
        return {
            (int(row["run_number"]), row["image_id"])
            for row in reader
            if row.get("experiment_id") == experiment_id
            and row.get("status") == "success"
        }


def _append_csv(output_csv: Path, row: dict[str, Any]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = output_csv.exists()
    with output_csv.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _safe_image_id(image_id: str) -> str:
    return image_id.replace("/", "__")


def _write_attempt_json(
    *,
    config: ExperimentConfig,
    experiment_id: str,
    manifest_path: Path,
    manifest_hash: str,
    item: SubsetItem,
    run_number: int,
    sequence_index: int,
    attempt: int,
    request_started_at: str,
    response_received_at: str,
    duration_seconds: float,
    status: str,
    provider_response: ProviderResponse | None,
    error: Exception | None,
) -> Path:
    output_path = (
        config.raw_json_dir
        / experiment_id
        / f"run_{run_number:02d}"
        / _safe_image_id(item.image_id)
        / f"attempt_{attempt:02d}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    error_response = error if isinstance(error, ProviderRequestError) else None
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "subset_manifest": str(manifest_path.resolve()),
        "subset_manifest_sha256": manifest_hash,
        "provider": config.provider,
        "requested_model": config.requested_model,
        "actual_model": (
            provider_response.actual_model
            if provider_response
            else getattr(error_response, "actual_model", "")
        ),
        "prompt": {
            "version": PROMPT_VERSION,
            "sha256": PROMPT_SHA256,
            "text": PROMPT,
        },
        "settings": {
            "temperature_requested": config.temperature_requested,
            "reasoning_mode": config.reasoning_mode,
            "provider_reasoning_setting": config.provider_reasoning_setting,
            "image_detail": config.image_detail,
        },
        "run_number": run_number,
        "sequence_index": sequence_index,
        "attempt": attempt,
        "image": {
            "image_id": item.image_id,
            "filename": item.filename,
            "path": str(item.image_path.resolve()),
            "selection_stratum": item.selection_stratum,
        },
        "request_started_at_utc": request_started_at,
        "response_received_at_utc": response_received_at,
        "duration_seconds": duration_seconds,
        "status": status,
        "parsed_result": (
            provider_response.parsed_result if provider_response else None
        ),
        "raw_output": (
            provider_response.raw_output
            if provider_response
            else getattr(error_response, "raw_output", "")
        ),
        "usage": (
            provider_response.usage
            if provider_response
            else getattr(error_response, "usage", {})
        ),
        "response": (
            provider_response.response_json
            if provider_response
            else getattr(error_response, "response_json", {})
        ),
        "error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
    }

    temporary_path = output_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(envelope, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    return output_path


def _run_order(
    items: list[SubsetItem],
    run_number: int,
    repeats: int,
) -> list[SubsetItem]:
    shift = ((run_number - 1) * math.ceil(len(items) / repeats)) % len(items)
    return items[shift:] + items[:shift]


def run_experiment(
    *,
    args: argparse.Namespace,
    config: ExperimentConfig,
    build_client: Callable[[], Any],
    request_once: Callable[[Any, Path], ProviderResponse],
) -> None:
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.request_delay < 0 or args.inter_run_delay < 0:
        raise ValueError("Delay values cannot be negative")

    items, manifest_hash = load_subset_manifest(
        args.subset_manifest,
        args.image_dir,
    )
    run_numbers = (
        sorted(set(args.run_numbers))
        if args.run_numbers
        else list(range(1, args.repeats + 1))
    )
    if any(run_number > args.repeats for run_number in run_numbers):
        raise ValueError(
            "Every --run-numbers value must be less than or equal to --repeats."
        )
    config = ExperimentConfig(
        provider=config.provider,
        requested_model=config.requested_model,
        output_csv=args.output_csv,
        raw_json_dir=args.raw_json_dir,
        temperature_requested=config.temperature_requested,
        reasoning_mode=config.reasoning_mode,
        provider_reasoning_setting=config.provider_reasoning_setting,
        image_detail=config.image_detail,
    )

    print("=" * 72)
    print(f"Experiment ID: {args.experiment_id}")
    print(f"Provider/model: {config.provider} / {config.requested_model}")
    print(f"Subset: {len(items)} fixed images ({manifest_hash[:12]}...)")
    print(f"Repeats: {args.repeats}")
    print(f"Runs selected now: {run_numbers}")
    print(f"Planned image-level requests now: {len(items) * len(run_numbers)}")
    print(f"Temperature: {config.temperature_requested}")
    print(f"Reasoning mode: {config.reasoning_mode}")
    print(f"Provider reasoning setting: {config.provider_reasoning_setting}")
    print(f"Output CSV: {config.output_csv.resolve()}")
    print(f"Raw JSON root: {config.raw_json_dir.resolve()}")
    print("=" * 72)

    if args.dry_run:
        print("Dry run complete: inputs and settings are valid; no API was called.")
        return

    completed = _completed_keys(config.output_csv, args.experiment_id)
    client = build_client()
    success_count = 0
    failure_count = 0

    for run_position, run_number in enumerate(run_numbers, start=1):
        ordered_items = _run_order(items, run_number, args.repeats)
        print(f"\n--- Run {run_number}/{args.repeats} ---")

        for sequence_index, item in enumerate(ordered_items, start=1):
            key = (run_number, item.image_id)
            if key in completed:
                print(
                    f"[{sequence_index}/{len(items)}] {item.image_id}: "
                    "already successful; skipped"
                )
                continue

            print(f"[{sequence_index}/{len(items)}] {item.image_id}")
            final_error: Exception | None = None

            for attempt in range(1, args.max_attempts + 1):
                request_started_at = utc_now()
                started = time.perf_counter()
                provider_response: ProviderResponse | None = None
                try:
                    provider_response = request_once(client, item.image_path)
                    response_received_at = utc_now()
                    duration_seconds = round(time.perf_counter() - started, 3)
                    raw_json_path = _write_attempt_json(
                        config=config,
                        experiment_id=args.experiment_id,
                        manifest_path=args.subset_manifest,
                        manifest_hash=manifest_hash,
                        item=item,
                        run_number=run_number,
                        sequence_index=sequence_index,
                        attempt=attempt,
                        request_started_at=request_started_at,
                        response_received_at=response_received_at,
                        duration_seconds=duration_seconds,
                        status="success",
                        provider_response=provider_response,
                        error=None,
                    )
                    result = provider_response.parsed_result
                    usage = provider_response.usage
                    _append_csv(
                        config.output_csv,
                        {
                            "schema_version": SCHEMA_VERSION,
                            "experiment_id": args.experiment_id,
                            "subset_manifest": str(args.subset_manifest.resolve()),
                            "subset_manifest_sha256": manifest_hash,
                            "selection_stratum": item.selection_stratum,
                            "run_number": run_number,
                            "sequence_index": sequence_index,
                            "image_id": item.image_id,
                            "filename": item.filename,
                            "image_path": str(item.image_path.resolve()),
                            "provider": config.provider,
                            "requested_model": config.requested_model,
                            "actual_model": provider_response.actual_model,
                            "prompt_version": PROMPT_VERSION,
                            "prompt_sha256": PROMPT_SHA256,
                            "temperature_requested": config.temperature_requested,
                            "reasoning_mode": config.reasoning_mode,
                            "provider_reasoning_setting": json.dumps(
                                config.provider_reasoning_setting,
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            "image_detail": config.image_detail,
                            "request_started_at_utc": request_started_at,
                            "response_received_at_utc": response_received_at,
                            "duration_seconds": duration_seconds,
                            "attempts_used": attempt,
                            "label": result["label"],
                            "confidence": result["confidence"],
                            "brief_reason": result["brief_reason"],
                            "status": "success",
                            "error": "",
                            "response_id": provider_response.response_id,
                            "input_tokens": usage.get("input_tokens", ""),
                            "output_tokens": usage.get("output_tokens", ""),
                            "reasoning_tokens": usage.get("reasoning_tokens", ""),
                            "total_tokens": usage.get("total_tokens", ""),
                            "raw_output": provider_response.raw_output,
                            "raw_json_path": str(raw_json_path.resolve()),
                        },
                    )
                    success_count += 1
                    completed.add(key)
                    print(
                        f"  {result['label']} ({result['confidence']}), "
                        f"actual model={provider_response.actual_model or 'unknown'}, "
                        f"{duration_seconds}s"
                    )
                    break
                except Exception as error:
                    final_error = error
                    response_received_at = utc_now()
                    duration_seconds = round(time.perf_counter() - started, 3)
                    _write_attempt_json(
                        config=config,
                        experiment_id=args.experiment_id,
                        manifest_path=args.subset_manifest,
                        manifest_hash=manifest_hash,
                        item=item,
                        run_number=run_number,
                        sequence_index=sequence_index,
                        attempt=attempt,
                        request_started_at=request_started_at,
                        response_received_at=response_received_at,
                        duration_seconds=duration_seconds,
                        status="failed",
                        provider_response=None,
                        error=error,
                    )
                    print(
                        f"  attempt {attempt}/{args.max_attempts} failed: "
                        f"{type(error).__name__}: {error}"
                    )
                    if attempt < args.max_attempts:
                        time.sleep(args.retry_base_delay * (2 ** (attempt - 1)))
            else:
                failure_count += 1
                error_response = (
                    final_error
                    if isinstance(final_error, ProviderRequestError)
                    else None
                )
                _append_csv(
                    config.output_csv,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "experiment_id": args.experiment_id,
                        "subset_manifest": str(args.subset_manifest.resolve()),
                        "subset_manifest_sha256": manifest_hash,
                        "selection_stratum": item.selection_stratum,
                        "run_number": run_number,
                        "sequence_index": sequence_index,
                        "image_id": item.image_id,
                        "filename": item.filename,
                        "image_path": str(item.image_path.resolve()),
                        "provider": config.provider,
                        "requested_model": config.requested_model,
                        "actual_model": getattr(error_response, "actual_model", ""),
                        "prompt_version": PROMPT_VERSION,
                        "prompt_sha256": PROMPT_SHA256,
                        "temperature_requested": config.temperature_requested,
                        "reasoning_mode": config.reasoning_mode,
                        "provider_reasoning_setting": json.dumps(
                            config.provider_reasoning_setting,
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        "image_detail": config.image_detail,
                        "request_started_at_utc": "",
                        "response_received_at_utc": utc_now(),
                        "duration_seconds": "",
                        "attempts_used": args.max_attempts,
                        "label": "",
                        "confidence": "",
                        "brief_reason": "",
                        "status": "failed",
                        "error": (
                            ""
                            if final_error is None
                            else f"{type(final_error).__name__}: {final_error}"
                        ),
                        "response_id": getattr(error_response, "response_id", ""),
                        "input_tokens": "",
                        "output_tokens": "",
                        "reasoning_tokens": "",
                        "total_tokens": "",
                        "raw_output": getattr(error_response, "raw_output", ""),
                        "raw_json_path": "",
                    },
                )

            time.sleep(args.request_delay)

        if run_position < len(run_numbers) and args.inter_run_delay:
            time.sleep(args.inter_run_delay)

    print("\n" + "=" * 72)
    print(f"New successes: {success_count}")
    print(f"Final failures: {failure_count}")
    print(f"CSV: {config.output_csv.resolve()}")
    print("=" * 72)
