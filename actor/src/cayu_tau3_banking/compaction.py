"""Faithful conversation compression, not an additional banking decision maker."""

from cayu import RetryPolicy
from cayu.context.base import ModelCompactor
from cayu.providers import ModelProvider, OpenAISubscriptionProvider
from cayu.providers.openai import OpenAIProvider

from cayu_tau3_banking.summary_provider import ReadOnlySummaryProvider

SUMMARY_INSTRUCTIONS = """Compress the supplied dialogue faithfully; do not solve the case.
Preserve the customer's objective using their exact words and source message ID
when available; keep later explicit changes with their scope. Do not substitute a
plan's goal, an agent's formula or a product's advertised benefits for that objective.
If an agent interpretation differs, label it as an agent interpretation rather than
attributing it to the customer. Do not settle that difference yourself.
For unresolved comparisons preserve the explored alternatives and the source facts
that distinguish them, including pair-specific benefits, costs, qualifications and
conflicting terms. Do not keep only the favored option or an attractive headline.
Prefer these distinctions over incidental features and long lists of lookup IDs.
Retained source-checked comparison facts are projected separately; do not duplicate
that worksheet or treat an omitted summary fact as a disproved or discarded option.
Preserve customer constraints, consent and refusals, pending handoffs, exact tool
result IDs, errors, amounts, resource identifiers and relevant public procedures.
Distinguish tool-observed facts, customer reports, agent conclusions, proposals and
unresolved questions. Attribute each to its source. Preserve uncertainty already
present in the dialogue without adding new uncertainty or resolving contradictions.
Do not introduce a prerequisite, prohibition, verification requirement or next step
that was absent from the source. Do not independently reassess eligibility,
recalculate a remedy, reverse an interpretation, or turn a suggestion into policy.
Preserve explicit status fields separately from dates and descriptions; do not
infer a different status from an apparent date inconsistency. Record both facts.
Preserve known prerequisites shared across pending outcomes, including capabilities
or eligibility an irreversible action would remove, only as stated in the source.
Label proposed ordering as proposed, not as a required procedure.
Do not invent successful actions. The fresh application evidence projection and
original sources remain authoritative. A summary cannot grant or withdraw consent,
prove completion, or create policy. Preserve material source locators so the acting
agent can revisit a disputed interpretation. Omit agent-authored plans, rejected
 drafts, proposed tool arguments, outcome packets, and stale “not ordered” or
“pending” claims unless they are direct tool-observed results. Retain verified
customer statements, successful tool results, explicit current requests or
refusals, unresolved questions, and source locators. Return only the summary,
aiming for at most 8000 characters. Do not call tools."""


def banking_compactor(provider: ModelProvider, model: str) -> ModelCompactor:
    """Retry only read-only summary inference, within Cayu's settlement safeguards."""
    return ModelCompactor(
        provider=ReadOnlySummaryProvider(provider)
        if isinstance(provider, (OpenAIProvider, OpenAISubscriptionProvider))
        else provider,
        model=model,
        options={"thinking": {"enabled": True, "effort": "low"}},
        max_input_chars=120_000,
        max_hierarchy_calls=8,
        retry_policy=RetryPolicy(max_attempts=4, max_unknown_attempts=2, initial_delay_s=2.0, max_delay_s=30.0),
        retry_empty_summaries=True,
        system_prompt=SUMMARY_INSTRUCTIONS,
    )
