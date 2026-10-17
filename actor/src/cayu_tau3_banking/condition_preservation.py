"""Source-complete, cross-outcome planning guidance for the acting model.

This is not a semantic validator: the deterministic controller checks references
and status transitions, not whether the model extracted every policy condition.
"""

LEGACY_CONDITION_PRESERVATION = """Before an irreversible change, consider its effects on EVERY pending requested outcome,
not only the local prerequisites of that action. Determine from policy and observed state
whether other requested work needs a capability, eligibility or resource this change would
remove. Preserve that prerequisite until the dependent work succeeds, a verified alternative
preserves it, or the customer explicitly changes the conflicting request. Record the resulting
cross-outcome dependencies; do not infer execution order from the customer's list order,
checklist position or a summary's suggested next steps. A valid local action can still make
the combined request impossible. If the necessary order is blocked, explain the conflict
and ask for the missing input or priority decision; do not silently abandon another outcome.
Do not invent dependent work when no other requested outcome needs the resource."""

CONDITION_PRESERVATION = """Before changing or retiring a resource shared by unfinished requested outcomes, check
the proposed AFTER state against the full source policy for EACH affected outcome.
In the existing obligation evidence, record the complete conjunction or alternatives
of material conditions, the particular resource satisfying them now, and what will
satisfy them after the change. A citation alone is not this check. Read the cited
source rather than trusting a shortened plan or summary. Keep qualifiers such as
scope, duration, prior history and capacity: satisfying one condition is not satisfying
all of them. A replacement does not inherit another resource's qualifications.
If an action would remove a required condition, keep the resource until dependent
work succeeds or a verified alternative meets EVERY required condition. Record that
ordering with cross-outcome dependencies. Use the existing obligations; do not create
a new outcome, empty action, reviewer call or repeated update for this check.
A preference for doing something first, or asking why that order is not possible,
is not informed permission to delay or abandon another requested outcome. Explain
the concrete consequence BEFORE the irreversible action and obtain the customer's
choice if their preferred order conflicts with the remaining work. After an informed
choice, honor it and record the affected outcome truthfully as deferred or declined,
not completed. Do not repeat a warning or consent question already answered.
This check concerns actual shared requirements, not invented dependencies: independent
actions and authorized retirement with no affected pending work can proceed normally.
The model is responsible for source coverage; the controller validates provenance
and plan consistency, not the truth or completeness of this interpretation."""


def replace_condition_preservation(instructions: str) -> str:
    """Replace exactly the known baseline for a controlled decision contrast."""
    if instructions.count(LEGACY_CONDITION_PRESERVATION) != 1:
        raise ValueError("Expected exactly one baseline condition-preservation paragraph")
    return instructions.replace(LEGACY_CONDITION_PRESERVATION, CONDITION_PRESERVATION, 1)
