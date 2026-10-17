import asyncio
import json

import pytest
from tau2.data_model.message import AssistantMessage, UserMessage
from test_runtime_banking import assembly, call, text, work

from cayu_tau3_banking.admission import BankingToolAdmissionGate
from cayu_tau3_banking.agent_memory import BankingMemoryRuntime
from cayu_tau3_banking.controller import (
    SYNC_WORK_PLAN_TOOL,
    UPDATE_WORK_PLAN_TOOL,
    BankingWorkflowController,
)
from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.ledger import ConversationLedger
from cayu_tau3_banking.product_navigation import (
    BROWSE_TOOL,
    BUDGET_LIMITS,
    COMPARISON_TOOL,
    READ_TOOL,
    RECOVERY_LANE,
    ProductNavigation,
    _fact_number,
)

ENTRY = "doc_savings_accounts_nimbus_001"
PRODUCT = "personal_savings/nimbus"
QUOTE = "Base APY is 3.25%. Annual fee is $20."


def document(path, identifier=ENTRY, content=QUOTE):
    (path / f"{identifier}.json").write_text(
        json.dumps(
            {
                "id": identifier,
                "title": "Nimbus account terms",
                "content": content,
            }
        )
    )


@pytest.fixture
def navigation(tmp_path):
    document(tmp_path)
    document(tmp_path, "doc_savings_accounts_nimbus_plus_001", "Nimbus Plus APY is 8.5%.")
    document(tmp_path, "doc_credit_cards_aster_001", "Aster annual fee is $15.")
    return ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))


def worksheet():
    return {
        "product_ids": [PRODUCT],
        "required_fields": ["base_apy", "annual_fee"],
        "facts": [
            {
                "product_id": PRODUCT,
                "field": field,
                "value": value,
                "entry_id": ENTRY,
                "revision": 1,
                "quote": QUOTE,
            }
            for field, value in [("base_apy", "3.25"), ("annual_fee", "20")]
        ],
        "calculations": [
            {
                "label": "declared annual net on 12000",
                "terms": [
                    {"product_id": PRODUCT, "field": "base_apy", "coefficient": "120"},
                    {"product_id": PRODUCT, "field": "annual_fee", "coefficient": "-1"},
                ],
            }
        ],
    }


def test_delivered_source_reuse_does_not_depend_on_research_quota(navigation):
    navigation.allow_delivered_reuse = True
    args = {"documents": [{"entry_id": ENTRY, "revision": 1}]}
    navigation.state["customer"] = "c"
    navigation.state["recovery"] = {"turn": 6, "total": 24}
    navigation.state["lanes"][READ_TOOL] = {
        "turn": BUDGET_LIMITS[READ_TOOL][0],
        "total": BUDGET_LIMITS[READ_TOOL][1],
    }
    before = navigation.snapshot()
    assert navigation.available("c", READ_TOOL)
    assert (
        navigation.begin_call(
            READ_TOOL,
            args,
            "c",
            delivered=lambda entry, revision: entry == ENTRY and revision == "1",
        )
        == "delivered_source_reuse"
    )
    assert navigation.snapshot() == before
    with pytest.raises(ValueError, match="budget exhausted"):
        navigation.begin_call(READ_TOOL, args, "c", delivered=lambda *_: False)
    with pytest.raises(ValueError):
        navigation.begin_call(READ_TOOL, {"documents": []}, "c", delivered=lambda *_: True)


def test_explicit_categories_products_reads_and_evidence(navigation):
    categories = navigation.browse({})
    assert {r["category"] for r in categories["entries"]} == {
        "personal_savings",
        "personal_credit_cards",
    }
    products = navigation.browse({"category": "personal_savings"})
    assert {r["product_id"] for r in products["entries"]} == {PRODUCT, PRODUCT + "_plus"}
    docs = navigation.browse({"product_id": PRODUCT})
    assert [r["entry_id"] for r in docs["entries"]] == [ENTRY]
    assert docs["directory_is_evidence"] is False
    read = asyncio.run(navigation.corpus.read_document(ENTRY, 1))
    assert read["text"] == QUOTE and read["text_complete"]
    assert read["evidence_ref"]["reference_id"] == ENTRY
    with pytest.raises(ValueError):
        asyncio.run(navigation.corpus.read_document(ENTRY, 2))
    with pytest.raises(ValueError):
        navigation.browse({"category": "personal_credit_cards", "product_id": PRODUCT})


def test_product_packet_scope_does_not_include_similar_names(navigation):
    assert navigation.resolve_read_documents({"product_id": PRODUCT}) == [
        {"entry_id": ENTRY, "revision": 1}
    ]
    assert navigation.resolve_read_documents({"product_id": PRODUCT + "_plus"}) == [
        {"entry_id": "doc_savings_accounts_nimbus_plus_001", "revision": 1}
    ]


def test_literal_facts_and_decimal_calculation(navigation):
    result = asyncio.run(navigation.check(worksheet(), lambda *_: True))
    assert result["declared_cells_complete"]
    assert result["calculation_results"] == [
        {"label": "declared annual net on 12000", "value": "370.00"}
    ]
    assert not result["semantic_entailment_checked"]
    assert not result["authorizes_completion"]
    args = worksheet()
    args.pop("calculations")
    args["facts"].pop()
    result = asyncio.run(navigation.check(args, lambda *_: True))
    assert result["missing"] == [{"product_id": PRODUCT, "field": "annual_fee"}]
    assert not result["declared_cells_complete"]


@pytest.mark.parametrize("value", ["12,345", "+12,345.50", "-12,345.50", "0", "3.25"])
def test_grouped_source_numbers_are_calculable_and_survive_restore(tmp_path, value):
    from decimal import Decimal

    quote = f"Published amount: {value}."
    document(tmp_path, content=quote)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    args = worksheet()
    args["required_fields"] = ["amount"]
    args["numeric_cells"] = [{"product_id": PRODUCT, "field": "amount"}]
    args["facts"] = [{**args["facts"][0], "field": "amount", "value": value, "quote": quote}]
    args["calculations"] = [
        {
            "label": "double",
            "terms": [{"product_id": PRODUCT, "field": "amount", "coefficient": "2"}],
        }
    ]
    result = asyncio.run(nav.check(args, lambda *_: True))
    assert result["declared_inputs_complete"]
    assert Decimal(result["calculation_results"][0]["value"]) == Decimal(value.replace(",", "")) * 2
    assert result["facts"][0]["value"] == value
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    args["facts"] = []
    args["reuse_checked_facts"] = True
    repeated = asyncio.run(restored.check(args, lambda *_: True))
    assert repeated["calculation_results"] == result["calculation_results"]


@pytest.mark.parametrize(
    "value",
    [
        "1,23",
        "12,34,567",
        "1,234,",
        "1,000%",
        "$1,000",
        "1e3",
        "NaN",
        "Infinity",
        "1,000.123456789",
        "12,345,678,901,234,567",
    ],
)
def test_ambiguous_or_unbounded_number_is_not_normalized(value):
    assert _fact_number(value) is None


@pytest.mark.parametrize("value", ["12", "345", "12345"])
def test_cannot_quote_only_part_of_grouped_number(tmp_path, value):
    quote = "Published amount: 12,345."
    document(tmp_path, content=quote)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    args = worksheet()
    args.pop("calculations")
    args["facts"] = [{**args["facts"][0], "value": value, "quote": quote}]
    with pytest.raises(ValueError, match="literally"):
        asyncio.run(nav.check(args, lambda *_: True))


def test_quote_rejection_returns_exact_repair_source_without_accepting_it(navigation):
    args = worksheet()
    before = navigation.snapshot()
    args["facts"][0]["quote"] = "Nimbus pays a 3.25% APY."
    with pytest.raises(ValueError) as rejected:
        asyncio.run(navigation.check(args, lambda *_: True))
    recovery = json.loads(str(rejected.value).split("(not a validated cell): ", 1)[1])
    assert recovery["source_excerpt"] == QUOTE
    assert recovery["excerpt_complete"] is True
    assert recovery["facts_accepted"] is False
    assert recovery["evidence_ref"]["reference_id"] == ENTRY
    assert navigation.snapshot() == before
    args["facts"][0]["quote"] = recovery["source_excerpt"]
    assert asyncio.run(navigation.check(args, lambda *_: True))["declared_inputs_complete"]


def test_repair_excerpt_is_bounded_and_marks_omitted_source(tmp_path):
    document(tmp_path, content="x" * 2000 + "\n" + QUOTE + "\n" + "z" * 2000)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    args = worksheet()
    args["facts"][0]["quote"] = "Nimbus pays a 3.25% APY."
    with pytest.raises(ValueError) as rejected:
        asyncio.run(nav.check(args, lambda *_: True))
    recovery = json.loads(str(rejected.value).split("(not a validated cell): ", 1)[1])
    assert QUOTE in recovery["source_excerpt"]
    assert len(recovery["source_excerpt"]) <= 1600
    assert recovery["excerpt_complete"] is False


def test_unread_source_is_not_exposed_in_repair(navigation):
    args = worksheet()
    args["facts"][0]["quote"] = "Nimbus pays a 3.25% APY."
    with pytest.raises(ValueError, match="Read the exact") as rejected:
        asyncio.run(navigation.check(args, lambda *_: False))
    assert "source_excerpt" not in str(rejected.value)


@pytest.mark.parametrize(
    "mutation",
    [
        "other_product",
        "invented_quote",
        "partial_number",
        "old_revision",
        "duplicate",
        "missing_calculation_fact",
    ],
)
def test_invalid_comparison_rejected(navigation, mutation):
    args = worksheet()
    fact = args["facts"][0]
    if mutation == "other_product":
        fact["entry_id"] = "doc_savings_accounts_nimbus_plus_001"
    elif mutation == "invented_quote":
        fact["quote"] = "Base APY is 3.25% for every customer forever."
    elif mutation == "partial_number":
        fact["value"] = "3"
    elif mutation == "old_revision":
        fact["revision"] = 2
    elif mutation == "duplicate":
        args["facts"].append(dict(fact))
    else:
        args["facts"].pop()
    with pytest.raises(ValueError):
        asyncio.run(navigation.check(args, lambda *_: True))


def test_directory_is_not_delivery(navigation):
    navigation.browse({"product_id": PRODUCT})
    with pytest.raises(ValueError, match="Read the exact"):
        asyncio.run(navigation.check(worksheet(), lambda *_: False))


def test_mixed_product_comparison_has_distinct_required_cells(navigation):
    args = worksheet()
    args["required_fields"] = []
    args["product_ids"].append("personal_credit_cards/aster")
    args["facts"] = [
        args["facts"][0],
        {
            "product_id": "personal_credit_cards/aster",
            "field": "card_fee",
            "value": "15",
            "entry_id": "doc_credit_cards_aster_001",
            "revision": 1,
            "quote": "Aster annual fee is $15.",
        },
    ]
    args["required_cells"] = [
        {"product_id": f["product_id"], "field": f["field"]} for f in args["facts"]
    ]
    args["calculations"][0]["terms"][1] = {
        "product_id": "personal_credit_cards/aster",
        "field": "card_fee",
        "coefficient": "-1",
    }
    result = asyncio.run(navigation.check(args, lambda *_: True))
    assert result["declared_cells_complete"]
    assert result["calculation_results"][0]["value"] == "375.00"
    args["required_cells"].pop()
    with pytest.raises(ValueError, match="every selected product"):
        asyncio.run(navigation.check(args, lambda *_: True))


def test_navigation_budgets_replay_without_replenishment(navigation):
    per_turn, total = BUDGET_LIMITS[BROWSE_TOOL]
    for _ in range(per_turn):
        navigation.begin_call(BROWSE_TOOL, {}, "customer-message:1")
    restored = ProductNavigation(navigation.corpus, navigation.state)
    with pytest.raises(ValueError, match="budget exhausted"):
        restored.begin_call(BROWSE_TOOL, {}, "customer-message:1")
    for customer in range(2, total // per_turn + 1):
        for _ in range(per_turn):
            restored.begin_call(BROWSE_TOOL, {}, f"customer-message:{customer}")
    with pytest.raises(ValueError, match="budget exhausted"):
        restored.begin_call(BROWSE_TOOL, {}, "customer-message:next")
    with pytest.raises(ValueError):
        ProductNavigation(navigation.corpus, {})
    with pytest.raises(ValueError):
        navigation.begin_call(BROWSE_TOOL, {"offset": True}, "customer-message:2")


def test_native_directory_read_and_comparison_stay_inside_cayu(tmp_path):
    document(tmp_path)
    agent, provider, captured = assembly(
        tmp_path,
        [
            call(SYNC_WORK_PLAN_TOOL, {"items": [work()]}),
            call(BROWSE_TOOL, {"product_id": PRODUCT}, "browse"),
            call(READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]}, "read"),
            call(COMPARISON_TOOL, worksheet(), "check"),
            call(UPDATE_WORK_PLAN_TOOL, {"items": [work(done=True, reference=ENTRY)]}, "done"),
            text("The declared net is 370.00 based on these terms."),
        ],
    )
    state = agent.get_init_state()
    try:
        reply, _ = agent.generate_next_message(
            UserMessage.text("Explain these account terms."), state
        )
        assert "370.00" in reply.content
        bank = captured["state"]
        assert bank.controller.has_delivered_knowledge_revision(ENTRY, "1")
        assert (
            sum(
                lane["total"]
                for lane in bank.memory.export_state()["navigation_state"]["lanes"].values()
            )
            + bank.memory.export_state()["navigation_state"]["recovery"]["total"]
            == 3
        )
        assert not bank.ledger.actions
        tool_parts = [
            part
            for request in provider.requests
            for message in request.messages
            for part in message.content
            if part.type == "tool_result"
        ]
        assert tool_parts
        assert all(not part.is_error for part in tool_parts)
    finally:
        state.stop()


def test_memory_session_replays_navigation_budget(navigation):
    runtime = BankingMemoryRuntime(navigation.corpus)
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Find the terms."))

    def session(history):
        return runtime.new_session(
            ledger=ledger,
            controller=BankingWorkflowController(ledger),
            admission_gate=BankingToolAdmissionGate(ledger),
            message_history=history,
            ledger_enabled=True,
            workflow_controller_enabled=False,
        )

    original = session(None)
    for _ in range(BUDGET_LIMITS[BROWSE_TOOL][0]):
        assert not asyncio.run(original.invoke(BROWSE_TOOL, {})).is_error
    history = [
        AssistantMessage.text("Continuing", raw_data={"cayu_tau3_memory": original.export_state()})
    ]
    restored = session(history)
    assert asyncio.run(restored.invoke(BROWSE_TOOL, {})).is_error
    assert (
        restored.export_state()["navigation_state"]["lanes"][BROWSE_TOOL]["total"]
        == BUDGET_LIMITS[BROWSE_TOOL][0]
    )
    ledger.observe_input(UserMessage.text("Now check another category."))
    assert not asyncio.run(restored.invoke(BROWSE_TOOL, {})).is_error


def test_discovery_cannot_starve_reads_or_validation(navigation):
    for _ in range(BUDGET_LIMITS[BROWSE_TOOL][0]):
        navigation.begin_call(BROWSE_TOOL, {}, "c1")
    assert not navigation.available("c1", BROWSE_TOOL)
    assert navigation.available("c1", READ_TOOL)
    assert navigation.available("c1", COMPARISON_TOOL)
    for _ in range(BUDGET_LIMITS[READ_TOOL][0]):
        navigation.begin_call(READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]}, "c1")
    assert navigation.remaining("c1")[READ_TOOL]["customer_calls"] == 0
    assert navigation.available("c1", READ_TOOL)  # Known-source recovery remains independent.
    assert navigation.remaining("c1")[RECOVERY_LANE]["customer_calls"] == 6
    assert navigation.available("c1", COMPARISON_TOOL)
    for _ in range(BUDGET_LIMITS[COMPARISON_TOOL][0]):
        navigation.begin_call(COMPARISON_TOOL, worksheet(), "c1")
    assert not navigation.available("c1", COMPARISON_TOOL)
    assert navigation.available("c2", COMPARISON_TOOL)


def test_legacy_budget_migration_and_snapshot_isolation(navigation):
    migrated = ProductNavigation(navigation.corpus, {"total": 48, "turn": 12, "customer": "c1"})
    assert all(not migrated.available("c2", name) for name in BUDGET_LIMITS)
    partial = ProductNavigation(navigation.corpus, {"total": 3, "turn": 2, "customer": "c1"})
    assert all(lane == {"total": 3, "turn": 2} for lane in partial.state["lanes"].values())
    saved = partial.snapshot()
    restored = ProductNavigation(navigation.corpus, saved)
    restored.begin_call(BROWSE_TOOL, {}, "c1")
    assert saved["lanes"][BROWSE_TOOL]["total"] == 3
    assert partial.state["lanes"][BROWSE_TOOL]["total"] == 3
    saved["lanes"][READ_TOOL]["turn"] = True
    with pytest.raises(ValueError):
        ProductNavigation(navigation.corpus, saved)


def test_locator_retention_restores_exact_metadata_without_replenishing_budget(navigation):
    navigation.begin_call(BROWSE_TOOL, {"category": "personal_savings"}, "c1")
    navigation.browse({"category": "personal_savings"})
    navigation.remember_document(ENTRY)
    saved = navigation.snapshot()
    restored = ProductNavigation(navigation.corpus, saved)
    assert PRODUCT in restored.context()
    assert ENTRY in restored.context()
    assert QUOTE not in restored.context()
    assert restored.remaining("c1") == navigation.remaining("c1")
    saved["products"].clear()
    assert PRODUCT in restored.state["products"]
    malformed = restored.snapshot()
    malformed["products"].append("invented")
    with pytest.raises(ValueError, match="locators"):
        ProductNavigation(navigation.corpus, malformed)


def test_version_two_migration_preserves_budget_and_adds_empty_locators(navigation):
    saved = navigation.snapshot()
    saved.pop("recovery")
    saved.pop("named_worksheets")
    saved.pop("worksheet")
    saved.pop("products")
    saved.pop("documents")
    saved["version"] = 2
    restored = ProductNavigation(navigation.corpus, saved)
    assert restored.state["version"] == 6
    assert restored.state["products"] == []
    assert restored.state["lanes"] == saved["lanes"]


def test_partial_worksheet_survives_restore_and_incremental_reuse(navigation):
    args = worksheet()
    args.pop("calculations")
    second = args["facts"].pop()
    result = asyncio.run(navigation.check(args, lambda *_: True))
    assert result["retained_for_compaction"]
    assert not result["declared_cells_complete"]
    saved = navigation.snapshot()
    restored = ProductNavigation(navigation.corpus, saved)
    saved["worksheet"]["arguments"]["facts"].clear()
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert QUOTE in context and '"missing_count": 1' in context
    assert len(context.encode()) < 14_000
    args["facts"] = [second]
    args["reuse_checked_facts"] = True
    result = asyncio.run(restored.check(args, lambda *_: True))
    assert result["declared_cells_complete"]
    assert len(result["facts"]) == 2


def test_reused_evidence_is_revalidated_and_failure_is_atomic(navigation):
    asyncio.run(navigation.check(worksheet(), lambda *_: True))
    before = navigation.snapshot()
    args = worksheet()
    args["facts"] = []
    args["reuse_checked_facts"] = True
    with pytest.raises(ValueError, match="Read the exact"):
        asyncio.run(navigation.check(args, lambda *_: False))
    assert navigation.snapshot() == before
    args["required_fields"] = ["unrelated"]
    with pytest.raises(ValueError, match="cannot remove"):
        asyncio.run(navigation.check(args, lambda *_: True))
    assert navigation.snapshot() == before


def test_restore_does_not_trust_fabricated_quotes(navigation):
    asyncio.run(navigation.check(worksheet(), lambda *_: True))
    saved = navigation.snapshot()
    saved["worksheet"]["arguments"]["facts"][0].pop("source_span")
    saved["worksheet"]["arguments"]["facts"][0]["quote"] = "Invented 3.25 rate"
    restored = ProductNavigation(navigation.corpus, saved)
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert "invalidated" in context and "Invented" not in context
    assert restored.snapshot()["worksheet"] is None


def test_new_scope_replaces_facts_and_old_corpus_invalidates(navigation):
    asyncio.run(navigation.check(worksheet(), lambda *_: True))
    saved = navigation.snapshot()
    saved["worksheet"]["corpus_sha256"] = "different"
    assert ProductNavigation(navigation.corpus, saved).state["worksheet"] is None
    args = worksheet()
    args.pop("calculations")
    args["facts"] = []
    result = asyncio.run(navigation.check(args, lambda *_: True))
    assert len(result["missing"]) == 2
    assert navigation.state["worksheet"]["arguments"]["facts"] == []


def test_large_worksheet_rejects_atomically_instead_of_discarding_saved_facts(navigation):
    asyncio.run(navigation.check(worksheet(), lambda *_: True))
    before = navigation.snapshot()
    readiness = navigation.comparison_readiness()
    navigation.MAX_WORKSHEET_BYTES = 10
    with pytest.raises(ValueError, match="previous worksheet is unchanged"):
        asyncio.run(navigation.check(worksheet(), lambda *_: True))
    assert navigation.snapshot() == before
    assert navigation.comparison_readiness() == readiness


def test_large_comparisons_retain_evidence_without_expanding_context(tmp_path):
    for index in range(12):
        document(tmp_path, f"doc_savings_accounts_item{index:02}_001", QUOTE)
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    args = {
        "product_ids": [r["product_id"] for r in nav.records],
        "required_fields": ["base_apy", "annual_fee"],
        "facts": [
            {
                "product_id": r["product_id"],
                "field": field,
                "value": value,
                "entry_id": r["entry_id"],
                "revision": 1,
                "quote": QUOTE,
                "interpretation": "Customer applicability remains to be assessed. " * 8,
            }
            for r in nav.records
            for field, value in [("base_apy", "3.25"), ("annual_fee", "20")]
        ],
    }
    for name in ("one", "two", "three", "four"):
        result = asyncio.run(nav.check({**args, "worksheet_id": name}, lambda *_: True))
        assert result["retained_for_compaction"]
    saved = nav.snapshot()
    assert len(json.dumps(saved["named_worksheets"]["one"]).encode()) > 12_000
    restored = ProductNavigation(nav.corpus, saved)
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert len(context.encode()) <= nav.MAX_COMPARISON_CONTEXT_BYTES
    assert context.count('"context_omitted": true') == 4
    assert restored.snapshot() == saved
    for name in ("one", "two", "three", "four"):
        recovered = asyncio.run(
            restored.check({"worksheet_id": name, "facts": []}, lambda *_: True)
        )
        assert len(recovered["facts"]) == 24
        assert recovered["declared_cells_complete"]
        assert all(f["quote"] == QUOTE for f in recovered["facts"])


def test_oversized_restored_worksheet_is_rejected(navigation):
    asyncio.run(navigation.check(worksheet(), lambda *_: True))
    saved = navigation.snapshot()
    saved["worksheet"]["arguments"]["facts"][0]["interpretation"] = "x" * 64_000
    with pytest.raises(ValueError, match="Invalid retained comparison"):
        ProductNavigation(navigation.corpus, saved)


def test_version_three_migrates_without_inventing_comparison(navigation):
    saved = navigation.snapshot()
    saved.pop("recovery")
    saved.pop("named_worksheets")
    saved.pop("worksheet")
    saved["version"] = 3
    restored = ProductNavigation(navigation.corpus, saved)
    assert restored.state["worksheet"] is None
    assert restored.state["lanes"] == saved["lanes"]


def test_two_scopes_cover_twenty_products_without_overwriting(tmp_path):
    for category in ("savings_accounts", "business_checking_accounts"):
        for index in range(10):
            document(tmp_path, f"doc_{category}_item{index:02}_001", "Annual fee is $1.")
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    for scope in ("personal_savings", "business_checking"):
        records = [r for r in nav.records if r["category"] == scope]
        assert len(records) == 10
        args = {
            "worksheet_id": scope,
            "product_ids": [r["product_id"] for r in records],
            "required_fields": ["fee"],
            "facts": [
                {
                    "product_id": r["product_id"],
                    "field": "fee",
                    "value": "1",
                    "entry_id": r["entry_id"],
                    "revision": 1,
                    "quote": "Annual fee is $1.",
                }
                for r in records
            ],
        }
        assert asyncio.run(nav.check(args, lambda *_: True))["retained_for_compaction"]
    restored = ProductNavigation(nav.corpus, nav.snapshot())
    assert len(restored.state["named_worksheets"]) == 2
    for scope, saved in restored.state["named_worksheets"].items():
        args = {**saved["arguments"], "facts": [], "reuse_checked_facts": True}
        result = asyncio.run(restored.check(args, lambda *_: True))
        assert len(result["facts"]) == 10 and result["declared_cells_complete"]
        assert result["worksheet_id"] == scope
    context = asyncio.run(restored.comparison_context(lambda *_: True))
    assert context.count("Retained source-checked comparison") == 2
    assert len(context.encode()) < 26_000


def test_named_worksheet_bounds_and_isolated_invalidation(navigation):
    for name in ("default", "one", "two", "three"):
        args = {**worksheet(), "worksheet_id": name}
        assert asyncio.run(navigation.check(args, lambda *_: True))["retained_for_compaction"]
    before = navigation.snapshot()
    with pytest.raises(ValueError, match="previous worksheet is unchanged"):
        asyncio.run(navigation.check({**worksheet(), "worksheet_id": "four"}, lambda *_: True))
    assert navigation.snapshot() == before
    navigation.state["named_worksheets"]["one"]["arguments"]["facts"][0]["quote"] = "fake 3.25"
    context = asyncio.run(navigation.comparison_context(lambda *_: True))
    assert "one invalidated" in context
    assert set(navigation.state["named_worksheets"]) == {"two", "three"}
    assert navigation.state["worksheet"] is not None


def test_version_four_migration_preserves_checked_default(navigation):
    asyncio.run(navigation.check(worksheet(), lambda *_: True))
    saved = navigation.snapshot()
    saved.pop("recovery")
    saved["version"] = 4
    saved.pop("named_worksheets")
    restored = ProductNavigation(navigation.corpus, saved)
    assert restored.state["worksheet"] == saved["worksheet"]
    assert restored.state["named_worksheets"] == {}


def test_four_named_scopes_and_aggregate_byte_bound(navigation):
    for name in ("one", "two", "three", "four"):
        assert asyncio.run(
            navigation.check({**worksheet(), "worksheet_id": name}, lambda *_: True)
        )["retained_for_compaction"]
    assert (
        len(ProductNavigation(navigation.corpus, navigation.snapshot()).state["named_worksheets"])
        == 4
    )
    navigation.MAX_ALL_WORKSHEETS_BYTES = 100
    before = navigation.snapshot()
    with pytest.raises(ValueError, match="previous worksheet is unchanged"):
        asyncio.run(navigation.check({**worksheet(), "worksheet_id": "one"}, lambda *_: True))
    assert navigation.snapshot() == before
    assert set(navigation.state["named_worksheets"]) == {"one", "two", "three", "four"}


def test_memory_context_rebuilds_checked_quotes_without_transcript(navigation):
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Compare the products."))
    controller = BankingWorkflowController(ledger)
    runtime = BankingMemoryRuntime(navigation.corpus)

    def create(history):
        return runtime.new_session(
            ledger=ledger,
            controller=controller,
            admission_gate=BankingToolAdmissionGate(ledger),
            message_history=history,
            ledger_enabled=True,
            workflow_controller_enabled=False,
        )

    original = create(None)
    read = asyncio.run(
        original.invoke(READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]})
    )
    assert not read.is_error
    payload = json.loads(read.content)
    assert set(payload["navigation_budget"]) == {"customer", "lanes", "recovery", "charged_lane"}
    assert not asyncio.run(original.invoke(COMPARISON_TOOL, worksheet())).is_error
    restored = create(
        [
            AssistantMessage.text(
                "Summary omitted all comparison facts.",
                raw_data={"cayu_tau3_memory": original.export_state()},
            )
        ]
    )
    context = asyncio.run(restored.runtime_context())
    assert "Retained source-checked comparison" in context
    assert QUOTE in context
    assert restored.navigation.state["lanes"][READ_TOOL]["total"] == 1


def test_invalid_checker_ids_offer_exact_source_recovery_without_accepting_alias(navigation):
    args = worksheet()
    args["product_ids"] = ["Nimbus Savings"]
    for fact in args["facts"]:
        fact["product_id"] = "Nimbus Savings"
    with pytest.raises(ValueError) as error:
        asyncio.run(navigation.check(args, lambda *_: True))
    assert '"Nimbus Savings"' in str(error.value)
    assert PRODUCT in str(error.value)
    assert ENTRY in str(error.value)
    assert "No facts were validated" in str(error.value)
    assert asyncio.run(navigation.check(worksheet(), lambda *_: True))["declared_cells_complete"]


def test_locator_context_is_byte_bounded_and_reports_omissions(tmp_path):
    for i in range(100):
        document(tmp_path, f"doc_savings_accounts_product_{i:03d}_001")
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    for record in nav.records:
        nav.remember_document(record["entry_id"])
    assert len(nav.state["documents"]) == 64
    rendered = nav.context(max_bytes=1024)
    assert len(rendered.encode()) <= 1024
    body = json.loads(rendered.split("\n", 1)[1])
    assert body["omitted"]["documents"] > 0
    assert body["bounded_recent_locators"]
    assert not body["directory_is_evidence"]


def test_exact_read_returns_product_locator_and_context_keeps_it_after_discovery_exhaustion(
    tmp_path,
):
    document(tmp_path)
    memory = BankingMemoryRuntime(BankingKnowledgeCorpus.from_directory(tmp_path))
    ledger = ConversationLedger()
    ledger.observe_input(UserMessage.text("Explain the account."))
    session = memory.new_session(
        ledger=ledger,
        controller=BankingWorkflowController(ledger),
        admission_gate=BankingToolAdmissionGate(ledger),
        message_history=None,
        ledger_enabled=True,
        workflow_controller_enabled=False,
    )
    for _ in range(BUDGET_LIMITS[BROWSE_TOOL][0]):
        asyncio.run(session.invoke(BROWSE_TOOL, {}))
    result = asyncio.run(
        session.invoke(READ_TOOL, {"documents": [{"entry_id": ENTRY, "revision": 1}]})
    )
    assert not result.is_error
    assert json.loads(result.content)["documents"][0]["product_id"] == PRODUCT
    assert PRODUCT in asyncio.run(session.runtime_context())


def test_batch_discovery_preserves_scope_and_page_bounds(navigation):
    products = [PRODUCT, PRODUCT + "_plus"]
    result = navigation.browse({"product_ids": products})
    assert {row["product_id"] for row in result["entries"]} == set(products)
    assert not result["directory_is_evidence"]
    with pytest.raises(ValueError):
        navigation.browse({"product_id": PRODUCT, "product_ids": products})
    with pytest.raises(ValueError):
        navigation.browse({"category": "personal_credit_cards", "product_ids": products})
    with pytest.raises(ValueError):
        navigation.browse({"product_ids": [PRODUCT, "unknown"]})


def test_unknown_numeric_cell_is_not_complete(navigation):
    args = worksheet()
    args.pop("calculations")
    args["numeric_cells"] = [{"product_id": PRODUCT, "field": "annual_fee"}]
    args["facts"][1]["value"] = "Annual fee"
    result = asyncio.run(navigation.check(args, lambda *_: True))
    assert result["declared_cells_complete"]
    assert not result["declared_inputs_complete"]
    assert result["numeric_missing"] == args["numeric_cells"]
    args["facts"][1]["value"] = "20"
    assert asyncio.run(navigation.check(args, lambda *_: True))["declared_inputs_complete"]
    args["numeric_cells"][0]["field"] = "undeclared"
    with pytest.raises(ValueError, match="required cells"):
        asyncio.run(navigation.check(args, lambda *_: True))


def test_new_customer_refreshes_tool_exposure_before_model(navigation):
    from cayu_tau3_banking.runtime_banking import BankingRuntimeSession

    bank = BankingRuntimeSession(
        BankingMemoryRuntime(navigation.corpus), (), planning_mode="outcomes"
    )
    asyncio.run(bank.customer("Initial request"))
    customer = bank.ledger.customer_message_ids[-1]
    for _ in range(BUDGET_LIMITS[BROWSE_TOOL][0]):
        bank.memory.navigation.begin_call(BROWSE_TOOL, {}, customer)
    bank.refresh_exposure()
    assert BROWSE_TOOL not in bank.available_private_names
    assert READ_TOOL in bank.available_private_names
    assert COMPARISON_TOOL in bank.available_private_names
    asyncio.run(bank.customer("New customer request"))
    assert BROWSE_TOOL in bank.available_private_names
    assert (
        bank.memory.navigation.state["lanes"][BROWSE_TOOL]["total"] == BUDGET_LIMITS[BROWSE_TOOL][0]
    )


def test_698_document_split_terms_remain_readable_after_browsing(tmp_path):
    from cayu_tau3_banking.navigation_scale_probe import documents

    for identifier, title, content in documents():
        (tmp_path / f"{identifier}.json").write_text(
            json.dumps({"id": identifier, "title": title, "content": content})
        )
    nav = ProductNavigation(BankingKnowledgeCorpus.from_directory(tmp_path))
    assert len(nav.records) == 698
    products = [r["product_id"] for r in nav.browse({"category": "personal_savings"})["entries"]]
    assert len(products) == 8
    seen = []
    offset = 0
    while True:
        args = {"product_ids": products, "offset": offset}
        nav.begin_call(BROWSE_TOOL, args, "c1")
        page = nav.browse(args)
        assert len(page["entries"]) <= 32
        seen.extend(page["entries"])
        if page["next_offset"] is None:
            break
        offset = page["next_offset"]
    assert len(seen) == len({r["entry_id"] for r in seen}) == 56
    for _ in range(nav.remaining("c1")[BROWSE_TOOL]["customer_calls"]):
        nav.begin_call(BROWSE_TOOL, {}, "c1")
    assert not nav.available("c1", BROWSE_TOOL)
    for suffix in ("003", "005", "007"):
        targets = [r for r in seen if r["entry_id"].endswith(suffix)]
        for start in range(0, len(targets), 4):
            batch = targets[start : start + 4]
            nav.begin_call(
                READ_TOOL,
                {
                    "documents": [
                        {"entry_id": r["entry_id"], "revision": r["revision"]} for r in batch
                    ]
                },
                "c1",
            )
            for r in batch:
                assert asyncio.run(nav.corpus.read_document(r["entry_id"], r["revision"]))[
                    "text_complete"
                ]
    assert nav.available("c1", COMPARISON_TOOL)
    assert nav.state["lanes"][READ_TOOL]["total"] == 6
