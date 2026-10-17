"""Source selection and incremental case evidence; no benchmark-specific rules."""

import asyncio
import json
from copy import deepcopy

import pytest
from test_product_navigation import ENTRY, PRODUCT, QUOTE, document, worksheet

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.product_navigation import COMPARISON_TOOL, ProductNavigation


def make_navigation(tmp_path, content=QUOTE):
    document(tmp_path, content=content)
    return ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))


def compact_facts(args):
    for fact in args["facts"]:
        fact.pop("quote", None)
        fact["source_ref"] = {
            "kind": "knowledge_revision",
            "reference_id": fact.pop("entry_id"),
            "version": str(fact.pop("revision")),
        }
    return args


def check(nav, args, delivered=lambda *_: True):
    return asyncio.run(nav.check(args, delivered))


def test_source_handles_copy_exact_passage_and_keep_arithmetic(tmp_path):
    nav = make_navigation(tmp_path)
    args = compact_facts(worksheet())
    original = deepcopy(args)
    result = check(nav, args)
    assert args == original
    assert [f["quote"] for f in result["facts"]] == [QUOTE, QUOTE]
    assert result["calculation_results"][0]["value"] == "370.00"
    assert not result["semantic_entailment_checked"]
    assert not result["authorizes_completion"]
    assert all(f["evidence_ref"]["reference_id"] == ENTRY for f in result["facts"])


def test_repeated_value_in_one_complete_paragraph_keeps_all_qualifiers(tmp_path):
    paragraph = "Minimum balance is $20. If balance drops below $20, a fee applies."
    nav = make_navigation(tmp_path, paragraph)
    args = compact_facts(worksheet())
    args.pop("calculations")
    args["facts"] = [args["facts"][1]]
    result = check(nav, args)
    assert result["facts"][0]["quote"] == paragraph
    assert not result["semantic_entailment_checked"]


def test_saved_spans_reconstruct_exact_sources_without_repeating_quote_per_cell(tmp_path):
    paragraph = (
        "Annual fee is $20. " + "The exception requires a qualifying balance. " * 25
    ).strip()
    nav = make_navigation(tmp_path, paragraph)
    args = worksheet()
    args.pop("calculations")
    args["required_fields"] = [f"field-{i}" for i in range(12)]
    args["facts"] = [
        {**args["facts"][1], "field": field, "quote": paragraph}
        for field in args["required_fields"]
    ]
    result = check(nav, args)
    assert result["retained_for_compaction"]
    saved = nav.snapshot()
    assert all(
        "quote" not in f and "source_span" in f for f in saved["worksheet"]["arguments"]["facts"]
    )
    restored = ProductNavigation(nav.corpus, saved)
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert context.count(paragraph) == 1
    patch = {"worksheet_id": "default", "reuse_checked_facts": True, "facts": []}
    assert not check(restored, patch)["missing"]
    assert check(restored, patch)["facts"][0]["quote"] == paragraph


@pytest.mark.parametrize(
    "span",
    [
        {"start": -1, "end": 20},
        {"start": 0, "end": 999999},
        {"start": 10, "end": 5},
        {"start": True, "end": 20},
    ],
)
def test_source_spans_are_bounded_and_never_authorize_unknown_text(tmp_path, span):
    nav = make_navigation(tmp_path)
    args = compact_facts(worksheet())
    args["facts"][0]["source_span"] = span
    before = nav.snapshot()
    with pytest.raises(ValueError):
        check(nav, args)
    assert nav.snapshot() == before


def test_interpretation_is_retained_without_becoming_source_or_arithmetic(tmp_path):
    nav = make_navigation(tmp_path)
    args = compact_facts(worksheet())
    args["worksheet_id"] = "decision"
    args["facts"][0]["interpretation"] = "My estimate uses 99%, but the source value is unchanged."
    result = check(nav, args)
    assert result["calculation_results"][0]["value"] == "370.00"
    assert result["facts"][0]["interpretation"] == args["facts"][0]["interpretation"]
    assert result["facts"][0]["quote"] == QUOTE
    assert result["semantic_entailment_checked"] is False
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    second = check(restored, {"worksheet_id": "decision", "reuse_checked_facts": True, "facts": []})
    assert second["facts"] == result["facts"]
    assert second["authorizes_completion"] is False


@pytest.mark.parametrize("quote", [None, "A made-up paraphrase of the policy."])
def test_failed_quote_selection_returns_source_without_reread_or_state_change(tmp_path, quote):
    nav = make_navigation(tmp_path, "No annual fee applies.")
    args = compact_facts(worksheet())
    args.pop("calculations")
    fact = {**args["facts"][0], "value": "0", "interpretation": "I infer a zero annual fee."}
    if quote is not None:
        fact["quote"] = quote
    args["facts"] = [fact]
    before = nav.snapshot()
    with pytest.raises(ValueError) as failure:
        check(nav, args)
    recovery = json.loads(str(failure.value).split("(not a validated cell): ", 1)[1])
    assert recovery["source_excerpt"] == "No annual fee applies."
    assert recovery["excerpt_complete"] and recovery["omitted_characters"] == 0
    assert recovery["facts_accepted"] is False
    assert nav.snapshot() == before
    assert not nav.comparison_readiness()
    with pytest.raises(ValueError, match="Read the exact source") as denied:
        check(nav, args, delivered=lambda *_: False)
    assert "No annual fee applies" not in str(denied.value)


def test_quote_recovery_is_bounded_and_reports_omissions(tmp_path):
    text = "Long introduction. " * 200 + "\n\nAnnual fee is $20.\n\nTransfer fee is $20."
    nav = make_navigation(tmp_path, text)
    args = compact_facts(worksheet())
    args.pop("calculations")
    args["facts"] = [{**args["facts"][0], "value": "20"}]
    with pytest.raises(ValueError) as failure:
        check(nav, args)
    recovery = json.loads(str(failure.value).split("(not a validated cell): ", 1)[1])
    assert len(recovery["source_excerpt"]) <= 1600
    assert recovery["excerpt_complete"] is False
    assert recovery["omitted_characters"] == len(text) - len(recovery["source_excerpt"])
    assert "Annual fee is $20." in recovery["source_excerpt"]


def test_minimal_patch_restores_scope_facts_calculations_and_coverage(tmp_path):
    nav = make_navigation(tmp_path)
    args = compact_facts(worksheet())
    args["worksheet_id"] = "decision"
    args["coverage"] = {
        "objective": "Compare annual net",
        "alternatives": [
            {"product_ids": [PRODUCT], "status": "unresolved", "reason": "Need customer usage"}
        ],
        "open_questions": ["Annual usage?"],
    }
    first = check(nav, args)
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    patch = {"worksheet_id": "decision", "reuse_checked_facts": True, "facts": []}
    assert restored.validators[COMPARISON_TOOL].is_valid(patch)
    second = check(restored, patch)
    assert second["facts"] == first["facts"]
    assert second["coverage"] == first["coverage"]
    assert second["calculation_results"] == first["calculation_results"]
    assert not second["declared_coverage_complete"]
    assert "Annual usage?" in asyncio.run(restored.comparison_context(lambda *_: True))


def test_additive_scope_reuses_sources_but_keeps_new_requirements_incomplete(tmp_path):
    nav = make_navigation(tmp_path)
    first = check(nav, {**worksheet(), "worksheet_id": "decision"})
    patch = {
        "worksheet_id": "decision",
        "reuse_checked_facts": True,
        "facts": [],
        "required_fields": [*worksheet()["required_fields"], "additional_fee"],
    }
    second = check(nav, patch)
    assert second["facts"] == first["facts"]
    assert not second["declared_cells_complete"]
    assert second["missing"] == [{"product_id": PRODUCT, "field": "additional_fee"}]
    patch.pop("required_fields")
    patch["facts"] = [{**worksheet()["facts"][1], "field": "additional_fee"}]
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    third = check(restored, patch)
    assert third["declared_cells_complete"]
    assert len(third["facts"]) == 3


@pytest.mark.parametrize(
    "content,value",
    [
        ("Annual fee is $20.\n\nTransfer fee is $20.", "20"),
        ("Published amount: 12,345.", "345"),
        ("No annual fee applies.", "0"),
        ("x" * 1700 + " Annual fee is $20.", "20"),
    ],
)
def test_ambiguous_inferred_partial_and_oversized_values_reject_atomically(
    tmp_path, content, value
):
    nav = make_navigation(tmp_path, content)
    args = compact_facts(worksheet())
    args.pop("calculations")
    args["facts"] = [{**args["facts"][0], "value": value}]
    before = nav.snapshot()
    with pytest.raises(ValueError):
        check(nav, args)
    assert nav.snapshot() == before
    assert nav.comparison_readiness() == {}


def test_explicit_quote_disambiguates_without_silent_correction(tmp_path):
    quote = "Annual fee is $20."
    nav = make_navigation(tmp_path, quote + "\n\nTransfer fee is $20.")
    args = compact_facts(worksheet())
    args.pop("calculations")
    args["facts"] = [{**args["facts"][1], "quote": quote}]
    assert check(nav, args)["facts"][0]["quote"] == quote
    before = nav.snapshot()
    args["facts"][0]["quote"] = "The annual fee equals 20 dollars."
    with pytest.raises(ValueError, match="literally"):
        check(nav, args)
    assert nav.snapshot() == before


def test_failed_revalidation_never_normalizes_persisted_facts_in_place(tmp_path):
    nav = make_navigation(tmp_path)
    check(nav, {**worksheet(), "worksheet_id": "decision"})
    saved = nav.snapshot()
    compact_facts(saved["named_worksheets"]["decision"]["arguments"])
    saved["named_worksheets"]["decision"]["arguments"]["facts"][1]["source_ref"]["version"] = "2"
    restored = ProductNavigation(nav.corpus, saved)
    before = restored.snapshot()
    with pytest.raises(ValueError):
        check(restored, {"worksheet_id": "decision", "reuse_checked_facts": True, "facts": []})
    assert restored.snapshot() == before


def test_full_paragraph_preserves_conditions_and_table_rows(tmp_path):
    passage = "Only accounts aged 90 days qualify.\n| Tier | APY |\n| Base | 3.25% |"
    nav = make_navigation(tmp_path, "Intro.\n\n" + passage + "\n\nEnd.")
    args = compact_facts(worksheet())
    args.pop("calculations")
    args["facts"] = args["facts"][:1]
    assert check(nav, args)["facts"][0]["quote"] == passage


def test_changed_inputs_reopen_saved_conclusions_without_erasing_facts(tmp_path):
    nav = make_navigation(tmp_path)
    initial = worksheet()
    initial.update(
        worksheet_id="decision",
        coverage={
            "objective": "Compare annual net",
            "alternatives": [
                {
                    "product_ids": [PRODUCT],
                    "status": "compared",
                    "reason": "Compared using known inputs",
                }
            ],
            "open_questions": [],
        },
    )
    assert check(nav, initial)["declared_coverage_complete"]
    patch = {
        "worksheet_id": "decision",
        "reuse_checked_facts": True,
        "facts": [],
        "calculations": deepcopy(initial["calculations"]),
    }
    patch["calculations"][0]["terms"][0]["coefficient"] = "60"
    result = check(nav, patch)
    assert result["declared_inputs_complete"]
    assert not result["declared_coverage_complete"]
    assert result["coverage"]["alternatives"][0]["status"] == "unresolved"
    assert result["coverage"]["open_questions"]
    assert len(result["facts"]) == 2
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    asyncio.run(restored.comparison_context(lambda *_: True))
    assert restored.comparison_readiness() == {"decision": False}
    patch["coverage"] = initial["coverage"]  # An explicit model reassessment, not semantic proof.
    assert check(restored, patch)["declared_coverage_complete"]


@pytest.mark.parametrize(
    "mode", ["undelivered", "wrong_revision", "mixed_identity", "digest", "scope"]
)
def test_patches_cannot_bypass_source_and_scope_checks(tmp_path, mode):
    nav = make_navigation(tmp_path)
    check(nav, {**worksheet(), "worksheet_id": "decision"})
    args = compact_facts(worksheet())
    args.update(worksheet_id="decision", reuse_checked_facts=True)
    if mode == "wrong_revision":
        args["facts"][0]["source_ref"]["version"] = "2"
    elif mode == "mixed_identity":
        args["facts"][0]["entry_id"] = ENTRY
    elif mode == "digest":
        args["facts"][0]["source_ref"]["digest"] = "fake"
    elif mode == "scope":
        args["required_fields"] = ["something_else"]
    before = nav.snapshot()
    with pytest.raises(ValueError):
        check(nav, args, lambda *_: mode != "undelivered")
    assert nav.snapshot() == before


def test_native_lossy_compaction_reuses_facts_and_accepts_changed_customer_objective(tmp_path):
    from cayu import ScriptedModelProvider, ToolResultPart
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.product_navigation import READ_TOOL

    document(tmp_path)
    summary = ScriptedModelProvider(
        [text("Earlier research; all details omitted.") for _ in range(30)]
    )
    policy = CheckpointCompactionContextPolicy(
        compactor=ModelCompactor(provider=summary, model="lossy-script"),
        compact_after_messages=2,
        max_user_turns=1,
    )
    initial = compact_facts(worksheet())
    initial.update(
        worksheet_id="decision",
        coverage={
            "objective": "Compare annual net for 12000",
            "alternatives": [
                {"product_ids": [PRODUCT], "status": "unresolved", "reason": "Confirm amount"}
            ],
            "open_questions": ["What is your intended balance?"],
        },
    )
    patch = {
        "worksheet_id": "decision",
        "reuse_checked_facts": True,
        "facts": [],
        "coverage": {
            "objective": "Compare annual net for 6000",
            "alternatives": [
                {"product_ids": [PRODUCT], "status": "compared", "reason": "Use updated balance"}
            ],
            "open_questions": [],
        },
        "calculations": [
            {
                "label": "updated net",
                "terms": [
                    {"product_id": PRODUCT, "field": "base_apy", "coefficient": "60"},
                    {"product_id": PRODUCT, "field": "annual_fee", "coefficient": "-1"},
                ],
            }
        ],
    }
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]}, "read"),
            call(COMPARISON_TOOL, initial, "save"),
            text("What balance should I use?"),
            call(COMPARISON_TOOL, patch, "patch"),
            text("The updated declared net is 175.00."),
        ],
        planning_mode="outcomes",
        advice_mode="lean",
        persistent=True,
        compaction=policy,
        callback_timeout_s=90,
        host_tools=[],
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text("Compare on a balance of 12000."), state)
        reply, _ = agent.generate_next_message(UserMessage.text("Actually use 6000."), state)
        assert reply.content == "The updated declared net is 175.00."
        assert summary.requests
        bank = captured["state"]
        saved = bank.memory.navigation.snapshot()["named_worksheets"]["decision"]["arguments"]
        assert saved["coverage"] == patch["coverage"]
        assert len(saved["facts"]) == 2
        assert all(f["source_span"] == {"start": 0, "end": len(QUOTE)} for f in saved["facts"])
        assert not bank.ledger.actions
        results = [
            p
            for r in provider.requests
            for m in r.messages
            for p in m.content
            if isinstance(p, ToolResultPart)
        ]
        assert all(not p.is_error for p in results)
        serialized = provider.requests[3].model_dump_json()
        assert "What is your intended balance?" in serialized
        assert "Retained source-checked comparison" in serialized
        assert (
            bank.memory.navigation.state["recovery"]["total"]
            + bank.memory.navigation.state["lanes"][READ_TOOL]["total"]
            == 1
        )
    finally:
        state.stop()
