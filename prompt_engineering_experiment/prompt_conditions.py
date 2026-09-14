"""Frozen prompt conditions for the visual-cue prompt experiment.

P0 is imported verbatim from the retained repeatability experiment. P1 and P2
insert one decision procedure before the unchanged confidence/output section.
P2 was drafted in an isolated analysis that was allowed to inspect only five
visual-intervention families outside the 16-image human-evaluation set:
16-1, 34-1, g-7, 12-1, and 32-2. No survey annotations were available to that
analysis.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from repeatability_subset_common import PROMPT as BASELINE_PROMPT


@dataclass(frozen=True)
class PromptCondition:
    condition_id: str
    version: str
    text: str
    role: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


P1_PROCEDURE = """
Structured decision procedure (perform internally):

1. Inventory the current image neutrally: visible people and objects, actions
   and interactions, conditions or consequences, setting and background, and
   clearly readable text.
2. Separate direct observations from interpretations. Do not turn a plausible
   interpretation into a visible fact.
3. Compare the strongest candidate, its closest plausible alternative, and
   "None of these / Not clearly moral in nature." For each, identify supporting
   evidence, conflicting or missing evidence, and any assumption it requires.
4. Select the category supported by the strongest direct and whole-scene
   evidence, focusing on the primary moral meaning rather than every possible
   theme.
5. Reject a candidate if it depends on an unsupported event, intention,
   relationship, identity, causal story, or background fact. If no category
   remains clearly supported, select the None category.
6. Set confidence from the clarity, consistency, and sufficiency of the visible
   evidence. Lower confidence when materially different interpretations remain
   plausible.

Do not reveal these internal steps or add output fields. Return only the
required JSON.
""".strip()


P2_PROCEDURE = """
Cue-guided decision procedure (perform internally):

1. Inventory the current image by cue channel: direct actions and interactions;
   visible conditions and consequences; setting and background; facial
   expression and posture; clearly readable text; group composition; and
   visibility limitations.
2. Weight cues by diagnostic value, not by visual salience or emotional
   vividness. Directly depicted conduct, conditions, or consequences carry the
   most weight. An ambiguous expression, pose, item of clothing, demographic
   feature, or generic setting is normally supporting evidence only. One
   unambiguous diagnostic cue may suffice; otherwise require convergence across
   independent cue channels.
3. Integrate each cue with the whole scene. Treat the background as direct
   evidence only when it visibly contains the relevant event or consequence;
   otherwise use it only as context. Treat a gesture as stronger when it is
   coordinated or repeated and supported by a consistent setting.
4. Use text only for its clearly legible literal content and check whether it
   agrees with the depicted scene. Text may identify a claimed context, rule,
   or topic, but does not by itself establish what happened, its moral valence,
   its legitimacy, or its effect on anyone.
5. Do not derive moral meaning from demographic makeup or identity alone. Mere
   co-presence does not establish conduct, treatment, commitment, vulnerability,
   or conflict. Treat blurred, cropped, occluded, missing, or unreadable content
   as unavailable; do not reconstruct it or infer why it is absent.
6. Compare the strongest candidate, its closest plausible alternative, and
   "None of these / Not clearly moral in nature." Discount the most ambiguous
   cue. If the preferred category then depends on a hidden story or a
   cue-to-label shortcut, lower confidence or select the None category.
   Confidence should track cue convergence and conflict.

Do not reveal these internal steps or add output fields. Return only the
required JSON.
""".strip()


def _insert_before_confidence(base: str, procedure: str) -> str:
    marker = "\nConfidence:\n"
    if base.count(marker) != 1:
        raise ValueError("Baseline prompt must contain one Confidence section.")
    return base.replace(marker, f"\n\n{procedure}\n\nConfidence:\n")


PROMPTS = {
    "p0": PromptCondition(
        condition_id="p0",
        version="p0-baseline-v2",
        text=BASELINE_PROMPT,
        role="retained baseline",
    ),
    "p1": PromptCondition(
        condition_id="p1",
        version="p1-structured-control-v1-2026-09-05",
        text=_insert_before_confidence(BASELINE_PROMPT, P1_PROCEDURE),
        role="structured-decision control",
    ),
    "p2": PromptCondition(
        condition_id="p2",
        version="p2-blind-nonoverlap-cues-v1-2026-09-05",
        text=_insert_before_confidence(BASELINE_PROMPT, P2_PROCEDURE),
        role="visual-cue-guided intervention",
    ),
}


def prompt_manifest() -> list[dict[str, str]]:
    return [
        {
            "condition_id": prompt.condition_id,
            "version": prompt.version,
            "sha256": prompt.sha256,
            "role": prompt.role,
        }
        for prompt in PROMPTS.values()
    ]

