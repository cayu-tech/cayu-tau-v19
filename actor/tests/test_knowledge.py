from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from cayu import (
    InMemoryEmbeddingKnowledgeStore,
    TextEmbedding,
    TextEmbeddingProvider,
    TextEmbeddingRequest,
    TextEmbeddingResult,
)

from cayu_tau3_banking.knowledge import (
    KNOWLEDGE_SEARCH_TOOL,
    BankingKnowledgeCorpus,
    BankingKnowledgeError,
)


def run(coroutine):
    return asyncio.run(coroutine)


def write_document(directory: Path, document_id: str, title: str, content: str) -> None:
    (directory / f"{document_id}.json").write_text(
        json.dumps({"id": document_id, "title": title, "content": content}),
        encoding="utf-8",
    )


def write_human_transfer_policy(directory: Path, document_id: str = "transfer_policy") -> None:
    write_document(
        directory,
        document_id,
        "Internal: Human Agent Transfer Reason Codes",
        (
            "Use transfer_to_human_agents with the highest applicable reason. "
            "TIER 1 (HIGHEST PRIORITY): "
            "customer_demands_after_unavailable_offer_refusal. "
            "TIER 2: unconfirmed_external_communication."
        ),
    )


class ConceptEmbeddingProvider(TextEmbeddingProvider):
    name = "concept-embedding-test"

    def __init__(self) -> None:
        self.calls: list[TextEmbeddingRequest] = []

    async def embed_texts(self, request: TextEmbeddingRequest) -> TextEmbeddingResult:
        self.calls.append(request)
        return TextEmbeddingResult(
            model=request.model,
            embeddings=[
                TextEmbedding(index=index, vector=concept_vector(text))
                for index, text in enumerate(request.texts)
            ],
        )


def concept_vector(text: str) -> list[float]:
    folded = text.casefold()
    promotion = any(
        term in folded
        for term in (
            "introductory incentive",
            "invitation-only",
            "statement credit",
            "welcome reward",
        )
    )
    return [1.0, 0.0] if promotion else [0.0, 1.0]


def test_explicit_semantic_mode_uses_only_semantic_ranks(tmp_path):
    write_document(
        tmp_path, "promotion", "Campaign", "An invited applicant earns statement credit."
    )
    write_document(tmp_path, "routing", "Routing", "Routing number information.")
    provider = ConceptEmbeddingProvider()
    memory = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=provider,
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )
    before = len(provider.calls)
    result = json.loads(run(memory.search("introductory incentive", mode="semantic")))
    assert result["retrieval"]["mode"] == "semantic"
    assert result["documents"][0]["entry_id"] == "promotion"
    assert len(provider.calls) == before + 1
    assert all(
        match["channel"] == "knowledge.semantic"
        for doc in result["documents"]
        for match in doc["channel_matches"]
    )
    hybrid = json.loads(run(memory.search("introductory incentive", mode="hybrid")))
    assert hybrid["retrieval"]["mode"] == "hybrid"


def test_semantic_pagination_retains_mode_and_exact_source_excerpt(tmp_path):
    for index in range(6):
        write_document(tmp_path, f"routing_{index}", "Routing", "# Routing\nRouting details.")
    memory = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=ConceptEmbeddingProvider(),
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )
    first = json.loads(run(memory.search("routing", mode="semantic", limit=4)))
    cursor = first["navigation"]["next_cursor"]
    second = json.loads(run(memory.search("routing", mode="semantic", limit=4, cursor=cursor)))
    assert len(first["documents"]) == 4
    assert len(second["documents"]) == 2
    assert not {d["entry_id"] for d in first["documents"]}.intersection(
        d["entry_id"] for d in second["documents"]
    )
    excerpt = first["documents"][0]["matching_excerpt"]
    assert excerpt["text"] == first["documents"][0]["text"][excerpt["start"] : excerpt["end"]]
    with pytest.raises(ValueError, match="cursor"):
        run(memory.search("routing", mode="hybrid", cursor=cursor))


def test_semantic_failure_reports_missing_source_without_lexical_results(tmp_path):
    write_document(tmp_path, "routing", "Routing", "Routing number information.")

    class FailingQueryProvider(ConceptEmbeddingProvider):
        async def embed_texts(self, request):
            if request.texts == ["routing"]:
                raise RuntimeError("unavailable")
            return await super().embed_texts(request)

    memory = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=FailingQueryProvider(),
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )
    result = json.loads(run(memory.search("routing", mode="semantic")))
    assert result["documents"] == []
    assert result["retrieval_bounds"]["sources_incomplete"]
    assert result["source_coverage"][0]["reason"] == "semantic_failed"


@pytest.mark.parametrize("mode", ["semantic", "hybrid"])
def test_explicit_embedding_modes_never_silently_fall_back(tmp_path, mode):
    memory = corpus(tmp_path)
    with pytest.raises(ValueError, match="configured embeddings"):
        run(memory.search("card", mode=mode))


def corpus(tmp_path: Path) -> BankingKnowledgeCorpus:
    write_document(
        tmp_path,
        "card_limits",
        "Card purchase limits",
        "The Sapphire card has a daily purchase limit of $4,500.",
    )
    write_document(
        tmp_path,
        "transfer_timing",
        "International transfer timing",
        "International transfers normally arrive in three to five business days.",
    )
    return BankingKnowledgeCorpus.from_directory(tmp_path)


def test_missing_document_directory_has_a_domain_error(tmp_path: Path) -> None:
    with pytest.raises(BankingKnowledgeError, match="document path is unavailable"):
        BankingKnowledgeCorpus.from_directory(tmp_path / "missing")


def test_public_capability_directory_and_exact_read_are_source_bound(tmp_path: Path) -> None:
    write_document(
        tmp_path, "odd_title", "Filing mechanics", "Read check_delivery_status_1234 first."
    )
    write_document(tmp_path, "other", "Unrelated", "No delivery capability here.")
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)
    directory = memory.public_capability_directory()
    assert directory is not None and "check_delivery_status_1234" in directory
    assert "Filing mechanics" not in directory
    content, refs = run(
        memory.search_with_evidence("check_delivery_status_1234", mode="capability")
    )
    result = json.loads(content)
    assert result["retrieval"]["mode"] == "capability"
    assert result["retrieval"]["embedding_model"] is None
    assert [item["entry_id"] for item in result["documents"]] == ["odd_title"]
    assert result["documents"][0]["text_complete"] is True
    assert [ref.reference_id for ref in refs] == ["odd_title"]
    excluded, refs = run(
        memory.search_with_evidence(
            "check_delivery_status_1234",
            mode="capability",
            exclude_entry_ids={"odd_title"},
        )
    )
    assert not json.loads(excluded)["documents"] and not refs
    unknown = json.loads(run(memory.search("invented_delivery_status_9876", mode="capability")))
    assert unknown["query"] is None and unknown["documents"] == []


def test_capability_index_is_deterministic_bounded_and_has_no_substring_match(
    tmp_path: Path,
) -> None:
    for index in range(6):
        write_document(tmp_path, f"doc_{index}", "Same capability", "read_package_status_1234")
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)
    payload = json.loads(run(memory.search("read_package_status_1234", mode="capability", limit=2)))
    assert payload["matched_documents"] == 6
    assert [item["entry_id"] for item in payload["documents"]] == ["doc_0", "doc_1"]
    payload = json.loads(run(memory.search("read_package_status", mode="capability")))
    assert payload["documents"] == []
    assert memory.public_capability_directory() == memory.public_capability_directory()


def test_empty_capability_directory_is_not_injected(tmp_path: Path) -> None:
    memory = corpus(tmp_path)
    assert memory.public_capability_directory() is None


def test_public_documents_are_loaded_as_revision_pinned_cayu_knowledge(tmp_path: Path) -> None:
    memory = corpus(tmp_path)

    assert memory.document_count == 2
    assert len(memory.corpus_sha256) == 64
    assert memory.titles == {
        "card_limits": "Card purchase limits",
        "transfer_timing": "International transfer timing",
    }
    assert memory.retrieval_mode == "lexical"
    assert memory.embedding_model is None
    assert memory.embedding_dimensions is None
    assert memory.embedding_representation is None

    entry = run(memory.store.get_entry("card_limits", access_scope=memory.scope))
    assert entry is not None
    assert entry.revision == 1
    assert entry.source_id == "card_limits"
    assert entry.metadata["tau2_commit"]
    assert entry.created_at.isoformat() == "2026-07-16T22:24:56+00:00"
    assert entry.updated_at == entry.created_at


def test_human_transfer_policy_is_selected_by_public_semantics_and_revision_pinned(
    tmp_path: Path,
) -> None:
    write_human_transfer_policy(tmp_path)
    write_document(
        tmp_path,
        "unrelated_transfer_article",
        "Sending bank transfers",
        "Use transfer_to_human_agents only when applicable.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(memory.human_transfer_policy_context_with_evidence())

    assert memory.human_transfer_policy_document_count == 1
    assert context is not None
    assert "<banking-human-transfer-policy>" in context
    assert '"selection":"deterministic_public_human_transfer_policy"' in context
    assert '"entry_id":"transfer_policy"' in context
    assert (
        '"evidence_ref":{"kind":"knowledge_revision",'
        '"reference_id":"transfer_policy","version":"1"}' in context
    )
    assert "unrelated_transfer_article" not in context
    assert len(evidence) == 1
    assert evidence[0].reference_id == "transfer_policy"
    assert evidence[0].version == "1"
    assert evidence[0].digest is not None


def test_human_transfer_policy_context_can_exclude_an_already_delivered_revision(
    tmp_path: Path,
) -> None:
    write_human_transfer_policy(tmp_path)
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.human_transfer_policy_context_with_evidence(exclude_entry_ids={"transfer_policy"})
    )

    assert context is None
    assert evidence == ()


def test_credit_card_dispute_dossier_requires_and_returns_the_complete_policy_family(
    tmp_path: Path,
) -> None:
    required = (
        ("filing", "Filing a Credit Card Transaction Dispute (Internal)"),
        ("card_number", "How to Find Your Rho-Bank Credit Card Numbers Online"),
        ("eligibility", "Provisional Credit Eligibility Guidelines (Internal)"),
        ("history", "Checking User Dispute History (Internal)"),
    )
    for document_id, title in required:
        write_document(tmp_path, document_id, title, f"Complete public policy for {title}.")
    write_document(tmp_path, "unrelated", "Debit Card Disputes", "Different policy family.")
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    content, evidence = run(memory.credit_card_dispute_policy_dossier_with_evidence())
    payload = json.loads(content)

    assert payload["selection"] == "deterministic_public_credit_card_dispute_policy_family"
    assert len(payload["policy_dossier_id"]) == 64
    assert [document["entry_id"] for document in payload["documents"]] == [
        "history",
        "filing",
        "card_number",
        "eligibility",
    ]
    assert [reference.reference_id for reference in evidence] == [
        "history",
        "filing",
        "card_number",
        "eligibility",
    ]

    context, context_evidence = run(
        memory.credit_card_dispute_policy_context_with_evidence(
            exclude_entry_ids={"filing", "eligibility"}
        )
    )
    assert context is not None
    assert "<banking-credit-card-dispute-policy>" in context
    assert context.count('"entry_id":"history"') == 1
    assert context.count('"entry_id":"card_number"') == 1
    assert '"entry_id":"filing"' not in context
    assert [reference.reference_id for reference in context_evidence] == [
        "history",
        "card_number",
    ]


def test_credit_card_dispute_dossier_fails_closed_when_a_policy_is_missing(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "eligibility",
        "Provisional Credit Eligibility Guidelines (Internal)",
        "Incomplete family.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    with pytest.raises(BankingKnowledgeError, match="family is incomplete"):
        run(memory.credit_card_dispute_policy_dossier_with_evidence())


@pytest.mark.parametrize("usage", ["travel", "everyday purchases", "everyday spending"])
def test_product_comparison_includes_reward_terms_for_customer_usage(tmp_path, usage):
    for family in ("amber", "jade"):
        write_document(
            tmp_path,
            f"doc_credit_cards_{family}_001",
            f"{family} terms",
            "Annual fee: $50. Purchase protection included. Credit limit: $120,000.",
        )
        write_document(
            tmp_path,
            f"doc_credit_cards_{family}_002",
            f"{family} rewards",
            "Cash back: 3% on qualifying categories, 1% on other purchases.",
        )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)
    context, refs = run(
        memory.product_comparison_context_with_evidence(
            f"I am looking for a credit card for {usage} with purchase protection."
        )
    )
    assert context is not None
    assert {ref.reference_id for ref in refs} == {
        f"doc_credit_cards_{family}_{sequence}"
        for family in ("amber", "jade")
        for sequence in ("001", "002")
    }


def test_product_comparison_dossier_covers_requested_facets_across_products(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_credit_cards_ecocard_001",
        "EcoCard: Apply for a Sustainable Rewards Credit Card",
        "Minimum credit score: 0. Foreign transaction fee: 1.0%.",
    )
    write_document(
        tmp_path,
        "doc_credit_cards_ecocard_008",
        "EcoCard: Making Payments and Managing Your Account",
        "Minimum monthly payment: 1.0%. Virtual card management: Yes.",
    )
    write_document(
        tmp_path,
        "doc_credit_cards_crypto-cash_back_001",
        "Crypto-Cash Back: Getting Started",
        "Virtual card management: Yes. Foreign transaction fee: 0%.",
    )
    write_document(
        tmp_path,
        "doc_credit_cards_crypto-cash_back_009",
        "Crypto-Cash Back: Eligibility",
        "Minimum credit score: 660. Minimum payment: 1.5%.",
    )
    write_document(
        tmp_path,
        "doc_credit_cards_virtual_card_management_001",
        "Virtual Card Management: Overview",
        "Virtual cards can be managed online.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "Which credit card options have a foreign transaction fee below 1.5%, "
            "a minimum payment below 1.5%, virtual cards, and accept my FICO score?"
        )
    )

    assert context is not None
    assert "<banking-product-comparison-dossier>" in context
    assert '"product_scope":"personal"' in context
    assert '"evidence_ref":{"kind":"knowledge_revision"' in context
    assert [reference.reference_id for reference in evidence] == [
        "doc_credit_cards_crypto-cash_back_001",
        "doc_credit_cards_ecocard_001",
        "doc_credit_cards_crypto-cash_back_009",
        "doc_credit_cards_ecocard_008",
    ]
    assert "doc_credit_cards_virtual_card_management_001" not in context


def test_account_referral_comparison_covers_rewards_and_qualifications_per_family(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_bank_accounts_bank_accounts_(general)_009",
        "Internal: Retrieving Customer Account Information",
        (
            "Use get_all_user_accounts_by_user_id_3847 to retrieve a customer's complete "
            "current bank and credit-card accounts, including opening dates."
        ),
    )
    write_document(
        tmp_path,
        "doc_business_checking_accounts_hunter_green_001",
        "Getting Started with Hunter Green Business Checking",
        (
            "Referrer reward: $175. Qualifying deposit: $10,000. "
            "Minimum relationship duration: 60 days."
        ),
    )
    write_document(
        tmp_path,
        "doc_business_checking_accounts_world_blue_001",
        "Getting Started with World Blue International Checking",
        (
            "To be eligible to refer, maintain checking status for 90 days. "
            "The referred business must deposit at least $25,000."
        ),
    )
    write_document(
        tmp_path,
        "doc_business_checking_accounts_world_blue_007",
        "World Blue Referral Bonuses",
        "Your bonus is $300 per successful referral. Annual limit: 12.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "Which business checking account should I use to maximize my referral "
            "bonus for a business depositing $30,000?"
        )
    )

    assert context is not None
    assert [reference.reference_id for reference in evidence] == [
        "doc_business_checking_accounts_hunter_green_001",
        "doc_business_checking_accounts_world_blue_001",
        "doc_business_checking_accounts_world_blue_007",
        "doc_bank_accounts_bank_accounts_(general)_009",
    ]
    assert "$25,000" in context
    assert "$300" in context
    assert "get_all_user_accounts_by_user_id_3847" in context


def test_generic_account_referral_comparison_uses_complete_checking_catalog(
    tmp_path: Path,
) -> None:
    documents = (
        (
            "doc_bank_accounts_bank_accounts_(general)_009",
            "Internal: Retrieving Customer Account Information",
            "Use get_all_user_accounts_by_user_id_3847 for complete customer accounts.",
        ),
        (
            "doc_bank_accounts_bank_accounts_(general)_047",
            "Internal: Checking Account Referral Program Terms and Restrictions",
            "At most two checking referral bonuses may complete in a rolling nine-day window.",
        ),
        (
            "doc_bank_accounts_bank_accounts_(general)_048",
            "FAQ: Checking Account Referrals",
            "Checking referral deposit windows vary by account type.",
        ),
        (
            "doc_checking_accounts_blue_account_010",
            "FAQ: Blue Account",
            (
                "The referrer earns $35 and the new member receives $30. A referral "
                "requires the new member to deposit at least $500 within 60 days and "
                "30 days of checking tenure."
            ),
        ),
        (
            "doc_checking_accounts_green_fee-free_account_007",
            "Green Fee-Free Account Referral Program",
            (
                "The referrer earns $20 and the new member receives $35. A referral "
                "requires the new member to deposit at least $300 within 60 days and "
                "30 days of checking tenure."
            ),
        ),
    )
    for document_id, title, content in documents:
        write_document(tmp_path, document_id, title, content)
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "I want to refer a friend to one of the accounts you offer. Which one should "
            "I choose for the best combined referral bonus if they deposit around $600?"
        )
    )

    assert context is not None
    assert '"product_scope":"personal_checking"' in context
    assert {reference.reference_id for reference in evidence} == {
        "doc_bank_accounts_bank_accounts_(general)_009",
        "doc_bank_accounts_bank_accounts_(general)_047",
        "doc_bank_accounts_bank_accounts_(general)_048",
        "doc_checking_accounts_blue_account_010",
        "doc_checking_accounts_green_fee-free_account_007",
    }
    assert "$35" in context
    assert "$30" in context
    assert "rolling nine-day window" in context


def test_product_comparison_dossier_skips_single_product_service_request(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_credit_cards_ecocard_001",
        "EcoCard: Apply for a Sustainable Rewards Credit Card",
        "Minimum credit score: 0. Foreign transaction fee: 1.0%.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "How do I get a referral link for my EcoCard credit card?"
        )
    )

    assert context is None
    assert evidence == ()


def test_product_comparison_dossier_prefers_explicit_personal_scope(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_credit_cards_ecocard_001",
        "EcoCard: Eligibility",
        "Minimum credit score: 0.",
    )
    write_document(
        tmp_path,
        "doc_business_credit_cards_launch_001",
        "Launch Business Card: Eligibility",
        "Minimum credit score: 0.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "Which credit card option accepts my credit score? I want a personal card, not a "
            "business card."
        )
    )

    assert context is not None
    assert '"product_scope":"personal"' in context
    assert [reference.reference_id for reference in evidence] == ["doc_credit_cards_ecocard_001"]
    assert "doc_business_credit_cards_launch_001" not in context


@pytest.mark.parametrize(
    "incidental",
    [
        "I will take my business elsewhere.",
        "I need a decision in three business days.",
        "I run a business but this is for groceries at home.",
    ],
)
def test_incidental_business_language_does_not_hide_personal_product_families(tmp_path, incidental):
    write_document(tmp_path, "doc_credit_cards_alpha_001", "Alpha: Fees", "Annual fee: 0.")
    write_document(tmp_path, "doc_business_credit_cards_beta_001", "Beta: Fees", "Annual fee: 0.")
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    context, refs = run(
        corpus.product_comparison_context_with_evidence(
            f"I am looking for a credit card with no annual fee. {incidental}"
        )
    )
    assert '"product_scope":"personal"' in context
    assert [ref.reference_id for ref in refs] == ["doc_credit_cards_alpha_001"]


@pytest.mark.parametrize(
    "description",
    ["business credit card", "credit card for my business", "credit card for our company"],
)
def test_explicit_business_product_intent_remains_on_business_catalog(tmp_path, description):
    write_document(tmp_path, "doc_credit_cards_alpha_001", "Alpha: Fees", "Annual fee: 0.")
    write_document(tmp_path, "doc_business_credit_cards_beta_001", "Beta: Fees", "Annual fee: 0.")
    corpus = BankingKnowledgeCorpus.from_directory(tmp_path)
    context, refs = run(
        corpus.product_comparison_context_with_evidence(
            f"Which card is best? I need a {description} with no annual fee."
        )
    )
    assert '"product_scope":"business"' in context
    assert [ref.reference_id for ref in refs] == ["doc_business_credit_cards_beta_001"]


def test_product_comparison_dossier_covers_personal_savings_families(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_savings_accounts_green_account_(savings)_001",
        "Green Account (savings): Rates and withdrawals",
        "4.0% APY, daily compounding, and 8 free withdrawals per month.",
    )
    write_document(
        tmp_path,
        "doc_savings_accounts_green_account_(savings)_003",
        "Green Account (savings): Eco relationship bonus",
        "An EcoCard relationship provides a bonus APY on this green account.",
    )
    write_document(
        tmp_path,
        "doc_savings_accounts_silver_plus_account_001",
        "Silver Plus Account specifications",
        "A $2,500 minimum balance, daily compounding, and 15 free withdrawals per month.",
    )
    write_document(
        tmp_path,
        "doc_savings_accounts_silver_plus_account_003",
        "Silver Plus Account ATM rebates",
        "Out-of-network ATM fees receive automatic rebates.",
    )
    write_document(
        tmp_path,
        "doc_savings_accounts_silver_plus_account_005",
        "Silver Plus Account FAQ",
        "3.0% APY and built-in savings goal tracking.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "I already have two checking accounts. Pick the one best savings account for "
            "daily compounding, frequent withdrawals, ATM fee rebates, relationship bonus "
            "APY, a manageable minimum balance, savings goals, and green features.",
            exclude_entry_ids={"doc_savings_accounts_silver_plus_account_005"},
        )
    )

    assert context is not None
    assert '"product_scope":"personal_savings"' in context
    assert [reference.reference_id for reference in evidence] == [
        "doc_savings_accounts_green_account_(savings)_001",
        "doc_savings_accounts_silver_plus_account_001",
        "doc_savings_accounts_green_account_(savings)_003",
        "doc_savings_accounts_silver_plus_account_003",
    ]
    assert "doc_savings_accounts_silver_plus_account_005" not in context


def test_product_comparison_dossier_skips_ambiguous_multi_account_request(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_checking_accounts_blue_account_001",
        "Blue Account",
        "No monthly fee.",
    )
    write_document(
        tmp_path,
        "doc_savings_accounts_silver_account_001",
        "Silver Account",
        "3.0% APY.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "Which checking account and savings account are the best options?"
        )
    )

    assert context is None
    assert evidence == ()


def test_product_comparison_dossier_recognizes_followup_account_constraints(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_checking_accounts_green_account_(checking)_001",
        "Green Account (checking)",
        "A personal checking account with an overdraft fee and foreign ATM fee.",
    )
    write_document(
        tmp_path,
        "doc_checking_accounts_purple_account_001",
        "Purple Account",
        "A personal checking account with ATM rebates, no foreign transaction fee, and a "
        "$3,750 minimum balance.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "For this personal checking account, I need international ATM rebates, no "
            "foreign transaction fee, a balance below $5,000, and no overdraft surprises."
        )
    )

    assert context is not None
    assert '"product_scope":"personal_checking"' in context
    assert {reference.reference_id for reference in evidence} == {
        "doc_checking_accounts_green_account_(checking)_001",
        "doc_checking_accounts_purple_account_001",
    }


def test_account_comparison_dossier_keeps_specific_out_of_network_atm_fee(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_checking_accounts_green_fee-free_account_001",
        "Green Fee-Free Account features",
        "Foreign and out-of-network ATM withdrawal fees are both $0.",
    )
    write_document(
        tmp_path,
        "doc_checking_accounts_purple_account_001",
        "Purple Account at a glance",
        "Foreign ATM withdrawal fee is $0 and operator rebates are available.",
    )
    write_document(
        tmp_path,
        "doc_checking_accounts_purple_account_012",
        "Out-of-network ATM fees for Purple Account",
        "Each out-of-network ATM withdrawal has a separate $2.50 bank fee.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    context, evidence = run(
        memory.product_comparison_context_with_evidence(
            "Which personal checking account costs least for out-of-network ATM "
            "withdrawals in foreign currency?"
        )
    )

    assert context is not None
    assert [reference.reference_id for reference in evidence] == [
        "doc_checking_accounts_green_fee-free_account_001",
        "doc_checking_accounts_purple_account_001",
        "doc_checking_accounts_purple_account_012",
    ]


def test_automatic_recall_injects_exact_revision_and_provenance(tmp_path: Path) -> None:
    memory = corpus(tmp_path)

    recalled = run(memory.recall("What is the Sapphire card purchase limit?"))

    assert recalled.record_ids[0] == "card_limits"
    assert recalled.injected_count >= 1
    assert recalled.context is not None
    assert '"entry_id":"card_limits"' in recalled.context
    assert '"entry_revision":1' in recalled.context
    assert "$4,500" in recalled.context
    assert recalled.source_coverage == (
        ("knowledge", "partial"),
        ("tau3_banking_bm25", "complete"),
    )


def test_internal_search_returns_bounded_cayu_recall_results(tmp_path: Path) -> None:
    memory = corpus(tmp_path)

    result = json.loads(run(memory.search("international transfer arrival", limit=1)))

    assert result["returned"] == 1
    assert result["documents"][0]["entry_id"] == "transfer_timing"
    assert result["documents"][0]["entry_revision"] == 1
    assert result["documents"][0]["evidence_ref"] == {
        "kind": "knowledge_revision",
        "reference_id": "transfer_timing",
        "version": "1",
    }
    assert "three to five business days" in result["documents"][0]["text"]
    assert result["retrieval"]["mode"] == "lexical"


def test_hybrid_cayu_recall_surfaces_semantic_promotion_without_shared_terms(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "elite_campaign",
        "Elite acquisition campaign",
        "Qualifying invited applicants earn a $5,000 statement credit.",
    )
    write_document(
        tmp_path,
        "account_routing",
        "Account routing details",
        "Use the routing number shown on the account statement.",
    )
    provider = ConceptEmbeddingProvider()

    memory = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=provider,
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )
    recalled = run(memory.recall("Which personal credit card bonus is best?"))
    searched = json.loads(run(memory.search("Which personal credit card bonus is best?", limit=1)))

    assert isinstance(memory.store, InMemoryEmbeddingKnowledgeStore)
    assert memory.retrieval_mode == "hybrid"
    assert memory.embedding_model == "concept-embedding"
    assert memory.embedding_dimensions == 2
    assert recalled.record_ids[0] == "elite_campaign"
    assert recalled.source_coverage == (
        ("knowledge", "complete"),
        ("tau3_banking_bm25", "complete"),
        ("tau3_banking_offer_facets", "complete"),
    )
    assert searched["documents"][0]["entry_id"] == "elite_campaign"
    assert searched["retrieval"] == {
        "embedding_dimensions": 2,
        "embedding_model": "concept-embedding",
        "embedding_representation": "cayu.tau3.banking.title-body.v1",
        "fusion_configuration": "cayu.tau3.discovery.v1",
        "mode": "hybrid",
    }
    assert len(provider.calls) == 5
    assert len(provider.calls[0].texts) == 2
    assert set(provider.calls[0].texts) == {
        "Account routing details\n\nUse the routing number shown on the account statement.",
        (
            "Elite acquisition campaign\n\n"
            "Qualifying invited applicants earn a $5,000 statement credit."
        ),
    }
    query_texts = [call.texts[0] for call in provider.calls[1:]]
    assert any("personal consumer credit-card products" in text for text in query_texts)
    assert any("invitation-only promotional offers" in text for text in query_texts)
    assert all(call.dimensions == 2 for call in provider.calls)


def test_hybrid_corpus_fails_closed_when_embedding_backfill_is_incomplete(
    tmp_path: Path,
) -> None:
    write_document(tmp_path, "one", "One", "One public document.")

    with pytest.raises(BankingKnowledgeError, match="backfill failed"):
        run(
            BankingKnowledgeCorpus.from_directory_with_embeddings(
                tmp_path,
                embedding_provider=ConceptEmbeddingProvider(),
                embedding_model="concept-embedding",
                embedding_dimensions=3,
            )
        )


def test_hybrid_lexical_fallback_matches_baseline_without_embedding_calls(tmp_path: Path) -> None:
    write_document(
        tmp_path, "routing", "Routing details", "Sapphire international routing number lookup."
    )
    write_document(
        tmp_path, "campaign", "Welcome campaign", "Invited applicants receive a statement credit."
    )
    provider = ConceptEmbeddingProvider()
    hybrid = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=provider,
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )
    baseline = BankingKnowledgeCorpus.from_directory(tmp_path)
    calls_before = len(provider.calls)

    content, evidence = run(
        hybrid.search_with_evidence(
            "Sapphire international routing",
            mode="lexical",
            limit=1,
        )
    )
    expected, expected_evidence = run(
        baseline.search_with_evidence(
            "Sapphire international routing",
            limit=1,
        )
    )

    assert json.loads(content) == json.loads(expected)
    assert evidence == expected_evidence
    assert evidence[0].reference_id == "routing"
    assert evidence[0].version == "1"
    assert len(provider.calls) == calls_before
    excluded, excluded_evidence = run(
        hybrid.search_with_evidence(
            "Sapphire international routing",
            mode="lexical",
            exclude_entry_ids={"routing"},
        )
    )
    assert json.loads(excluded)["documents"] == []
    assert excluded_evidence == ()
    assert len(provider.calls) == calls_before


@pytest.mark.parametrize("mode", ["unsupported", "", None, True, [], {}])
def test_explicit_search_rejects_unsupported_mode(tmp_path: Path, mode: object) -> None:
    with pytest.raises(ValueError, match="mode must"):
        run(corpus(tmp_path).search("routing", mode=mode))


def test_hybrid_offer_facet_lane_skips_unrelated_queries(tmp_path: Path) -> None:
    write_document(
        tmp_path,
        "account_routing",
        "Account routing details",
        "Use the routing number shown on the account statement.",
    )
    provider = ConceptEmbeddingProvider()
    memory = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=provider,
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )

    recalled = run(memory.recall("How do I locate account routing details?"))

    assert recalled.record_ids[0] == "account_routing"
    assert recalled.source_coverage == (
        ("knowledge", "complete"),
        ("tau3_banking_bm25", "complete"),
        ("tau3_banking_offer_facets", "complete"),
    )
    assert len(provider.calls) == 2


def test_hybrid_recall_bounds_a_large_combined_source_before_admission(
    tmp_path: Path,
) -> None:
    for index in range(16):
        write_document(
            tmp_path,
            f"large_policy_{index:02d}",
            f"Large banking policy {index:02d}",
            "Shared banking policy guidance. " + (chr(ord("a") + index) * 7_600),
        )
    memory = run(
        BankingKnowledgeCorpus.from_directory_with_embeddings(
            tmp_path,
            embedding_provider=ConceptEmbeddingProvider(),
            embedding_model="concept-embedding",
            embedding_dimensions=2,
        )
    )

    recalled = run(memory.recall("shared banking policy guidance"))

    assert recalled.candidate_count <= 16
    assert recalled.injected_count <= 6
    assert recalled.context is not None
    assert len(recalled.context.encode("utf-8")) <= 60_000


def test_internal_search_can_omit_documents_already_delivered_this_turn(tmp_path: Path) -> None:
    memory = corpus(tmp_path)

    result = json.loads(
        run(
            memory.search_with_evidence(
                "international transfer arrival",
                limit=2,
                exclude_entry_ids={"transfer_timing"},
            )
        )[0]
    )

    assert result["documents"] == []
    assert result["returned"] == 0


@pytest.mark.parametrize("body_repetitions", [20, 3000])
def test_native_and_bm25_lanes_agree_on_bounded_title_and_multibyte_preview(
    tmp_path: Path, body_repetitions: int
) -> None:
    write_document(
        tmp_path,
        "long_policy",
        "Routing policy " + "é" * 600,
        "routing instructions " + "界" * body_repetitions,
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    async def scenario():
        # Compare source records directly so a silent optional-lane omission
        # cannot make this cross-source identity contract appear to pass.
        from cayu.memory.recall import KnowledgeRecallSource

        from cayu_tau3_banking.knowledge import BankingBm25RecallSource

        entry = await memory.store.get_entry("long_policy", access_scope=memory.scope)
        situation = memory._situation("routing instructions")
        bm25 = BankingBm25RecallSource(memory.store, (entry,), corpus_sha256=memory.corpus_sha256)
        native = KnowledgeRecallSource(memory.store, max_record_bytes=8_000)
        left = await bm25.retrieve(situation)
        right = await native.retrieve(situation)
        assert left.records and right.records
        if left.records[0].identity == right.records[0].identity:
            assert left.records[0] == right.records[0]
        else:
            # The native store automatically chunks long documents. Distinct
            # entry/chunk identities must still agree on preview and title.
            assert right.records[0].representation == "chunk_text"
            assert left.records[0].text == right.records[0].text
            assert left.records[0].title == right.records[0].title
        assert left.records[0].text_complete == (body_repetitions == 20)
        assert len(left.records[0].text.encode("utf-8")) <= 8_000
        assert len(left.records[0].title.encode("utf-8")) <= 1024

    run(scenario())


def test_reward_policy_dossier_selects_complete_public_policy_family(tmp_path: Path) -> None:
    for document_id, title, content in (
        (
            "silver_rates",
            "Silver Rewards Card: How to Earn 4% Cash Back",
            "Earn 4% on qualifying travel and software.",
        ),
        (
            "silver_exclusions",
            "Silver Rewards Card: Exceptions and Exclusions",
            "Example excluded merchants earn the base rate.",
        ),
        (
            "silver_referral",
            "Silver Rewards Card: Referral Program - Earn Rewards",
            "Referral policy is not a transaction earning rule.",
        ),
        (
            "rounding",
            "Credit Card Rewards Points Rounding Policy",
            "Floor fractional points.",
        ),
        (
            "representation",
            "Internal: Credit Card Rewards - Points vs Cash Back Representation",
            "One point is one cent.",
        ),
    ):
        write_document(tmp_path, document_id, title, content)
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    dossier = json.loads(run(memory.reward_policy_dossier(["Silver Rewards Card"])))

    assert dossier["card_types"] == ["Silver Rewards Card"]
    assert len(dossier["policy_dossier_id"]) == 64
    assert {item["entry_id"] for item in dossier["documents"]} == {
        "representation",
        "rounding",
        "silver_exclusions",
        "silver_referral",
        "silver_rates",
    }
    assert all(item["entry_revision"] == 1 for item in dossier["documents"])
    assert all(
        item["evidence_ref"]
        == {
            "kind": "knowledge_revision",
            "reference_id": item["entry_id"],
            "version": "1",
        }
        for item in dossier["documents"]
    )
    assert all(item["text_complete"] is True for item in dossier["documents"])


def test_savings_interest_dossier_selects_observed_product_policy_families(
    tmp_path: Path,
) -> None:
    documents = (
        (
            "doc_savings_accounts_gold_account_001",
            "Gold Savings Account: Complete Guide",
            "The Gold Account earns 5.5% APY.",
        ),
        (
            "doc_savings_accounts_gold_account_014",
            "Gold Account: Credit Card APY Bonuses",
            "EcoCard provides a 0.6% APY bonus.",
        ),
        (
            "doc_savings_accounts_gold_account_012",
            "Gold Account: External transfers",
            "External transfers may take three days.",
        ),
        (
            "doc_savings_accounts_silver_account_001",
            "Silver Account guide",
            "The Silver Account earns 4% APY.",
        ),
        (
            "doc_checking_accounts_green_account_(checking)_001",
            "Green Account at a glance",
            "Boost a linked savings Gold Account by 0.75% APY.",
        ),
        (
            "doc_checking_accounts_purple_account_001",
            "Purple Account at a glance",
            "The linked Gold savings APY boost is 0.1%.",
        ),
        (
            "doc_checking_accounts_blue_account_001",
            "Blue Account at a glance",
            "The linked savings APY boost is 0.35%.",
        ),
        (
            "general-checking-pairings",
            "Linked Checking Account APY Boosts for Savings Accounts",
            "Only documented pairings qualify.",
        ),
        (
            "general-card-stacking",
            "Credit Card APY Bonuses: Stacking Policy",
            "Only the highest active card APY bonus applies.",
        ),
    )
    for document_id, title, content in documents:
        write_document(tmp_path, document_id, title, content)
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    content, evidence = run(
        memory.savings_interest_policy_dossier_with_evidence(
            savings_account_level="Gold Account",
            checking_account_levels=["Green Account", "Purple Account"],
            credit_card_types=["Platinum Rewards Card", "EcoCard"],
        )
    )
    dossier = json.loads(content)

    assert dossier["savings_account_level"] == "Gold Account"
    assert dossier["checking_account_levels"] == ["Green Account", "Purple Account"]
    assert dossier["credit_card_types"] == ["Platinum Rewards Card", "EcoCard"]
    assert {item["entry_id"] for item in dossier["documents"]} == {
        "doc_savings_accounts_gold_account_001",
        "doc_savings_accounts_gold_account_014",
        "doc_checking_accounts_green_account_(checking)_001",
        "doc_checking_accounts_purple_account_001",
        "general-checking-pairings",
        "general-card-stacking",
    }
    assert {item.reference_id for item in evidence} == {
        item["entry_id"] for item in dossier["documents"]
    }
    assert len(dossier["policy_dossier_id"]) == 64


def test_savings_interest_dossier_rejects_missing_observed_checking_policy(
    tmp_path: Path,
) -> None:
    write_document(
        tmp_path,
        "doc_savings_accounts_gold_account_001",
        "Gold Savings Account: Complete Guide",
        "The Gold Account earns 5.5% APY.",
    )
    memory = BankingKnowledgeCorpus.from_directory(tmp_path)

    with pytest.raises(BankingKnowledgeError, match="No savings-interest policy"):
        run(
            memory.savings_interest_policy_dossier_with_evidence(
                savings_account_level="Gold Account",
                checking_account_levels=["Missing Account"],
                credit_card_types=[],
            )
        )


def test_bm25_plus_baseline_does_not_fabricate_nonmatching_documents(tmp_path: Path) -> None:
    memory = corpus(tmp_path)

    result = json.loads(run(memory.search("international", limit=2)))

    assert [item["entry_id"] for item in result["documents"]] == ["transfer_timing"]
    assert result["returned"] == 1


def test_search_schema_is_internal_read_only_memory_tool(tmp_path: Path) -> None:
    schema = corpus(tmp_path).search_tool_schema()

    assert schema["name"] == KNOWLEDGE_SEARCH_TOOL
    assert schema["input_schema"]["properties"]["limit"]["maximum"] == 4
    assert schema["input_schema"]["properties"]["limit"]["default"] == 4
    assert "customer records" in schema["description"]


def test_corpus_rejects_filename_identity_mismatch(tmp_path: Path) -> None:
    (tmp_path / "wrong.json").write_text(
        json.dumps({"id": "right", "title": "Title", "content": "Content"}),
        encoding="utf-8",
    )

    with pytest.raises(BankingKnowledgeError, match="filename does not match"):
        BankingKnowledgeCorpus.from_directory(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"id": "doc", "title": "Title"},
        {"id": "doc", "title": "Title", "content": ""},
    ],
)
def test_corpus_rejects_malformed_public_documents(tmp_path: Path, payload: object) -> None:
    (tmp_path / "doc.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BankingKnowledgeError):
        BankingKnowledgeCorpus.from_directory(tmp_path)
