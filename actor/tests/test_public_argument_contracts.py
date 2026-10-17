import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from cayu import ToolCallPart

from cayu_tau3_banking.knowledge import BankingKnowledgeCorpus
from cayu_tau3_banking.public_argument_contracts import (
    SAVINGS_NAME_CONTRACT,
    PublicArgumentContract,
    PublicArgumentContracts,
)


def proposal(arguments, *, tool="open_bank_account_4821", wrapped=True):
    return ToolCallPart(
        tool_call_id="c1",
        tool_name="call_discoverable_agent_tool" if wrapped else tool,
        arguments={"agent_tool_name": tool, "arguments": json.dumps(arguments)}
        if wrapped
        else arguments,
    )


@pytest.fixture(scope="module")
def banking():
    return BankingKnowledgeCorpus.from_directory(
        Path(".vendor/tau2/data/tau2/domains/banking_knowledge/documents")
    )


@pytest.mark.parametrize("wrapped", [True, False])
@pytest.mark.parametrize(
    "value,allowed",
    [
        ("Violet Account", True),
        ("Violet Plus Account", True),
        ("Green Account (savings)", False),
        ("Violet Account (archive)", False),
        ("Violet Account\n", False),
        ("Bad\nViolet Account", False),
        ("VioletAccount", False),
        ("Violet", False),
        (None, False),
    ],
)
def test_public_format_without_renaming_or_assuming_semantic_correctness(
    banking, wrapped, value, allowed
):
    call = proposal(
        {"user_id": "private-user", "account_type": "savings", "account_class": value},
        wrapped=wrapped,
    )
    before = call.model_dump(mode="json")
    (decision,) = banking.public_argument_contracts.review([call])
    assert decision.allowed is allowed
    assert call.model_dump(mode="json") == before
    if not allowed:
        from cayu_tau3_banking.argument_contract import SOURCE_BINDING_RECOVERY

        assert "private-user" not in decision.reason
        assert SAVINGS_NAME_CONTRACT.quote in decision.reason
        assert "does not establish product identity" in decision.reason
        assert decision.diagnostic_code == "public_argument_contract"
        assert SOURCE_BINDING_RECOVERY in decision.reason


@pytest.mark.parametrize("account_type", ["checking", "business_checking", "business_savings"])
def test_personal_savings_rule_does_not_extend_to_other_types(banking, account_type):
    call = proposal({"account_type": account_type, "account_class": "Other naming convention"})
    assert banking.public_argument_contracts.review([call])[0].allowed


@pytest.mark.parametrize(
    "raw",
    [
        "[]",
        "null",
        '{"account_type":"savings","account_type":"checking"}',
        '{"x":NaN}',
        '{"x":Infinity}',
        '{"x":-Infinity}',
        '{"x":1e9999}',
        "{broken",
    ],
)
def test_reject_ambiguous_or_nonobject_wrapper_before_dispatch(banking, raw):
    call = proposal({})
    call.arguments["arguments"] = raw
    assert not banking.public_argument_contracts.review([call])[0].allowed


def test_public_source_digest_and_quote_are_checked(banking):
    c = SAVINGS_NAME_CONTRACT
    document = asyncio.run(banking.read_document(c.entry_id, c.revision))
    assert hashlib.sha256(document["text"].encode()).hexdigest() == c.text_sha256
    entry = SimpleNamespace(revision=1, text=document["text"])
    for changed in [
        replace(c, revision=2),
        replace(c, quote="invented"),
        replace(c, text_sha256="0" * 64),
    ]:
        with pytest.raises(ValueError, match="source changed"):
            PublicArgumentContracts({c.entry_id: entry}, [changed])
    PublicArgumentContracts({}, [c])  # Unrelated corpus has no banking rules.


def test_synthetic_canonical_values_similar_names_and_optional_overrides():
    source = "API catalogue values are nimbus_e7 and nimbus_e8; display name Nimbus Enterprise Vault. scope is optional."
    c = PublicArgumentContract(
        "provision",
        "catalogue",
        1,
        hashlib.sha256(source.encode()).hexdigest(),
        source,
        json.dumps(
            {
                "type": "object",
                "required": ["sku"],
                "properties": {"sku": {"enum": ["nimbus_e7", "nimbus_e8"]}},
            }
        ),
    )
    gate = PublicArgumentContracts({"catalogue": SimpleNamespace(revision=1, text=source)}, [c])
    for sku in ("nimbus_e7", "nimbus_e8", "Nimbus Enterprise Vault", "nimbus_e9"):
        call = proposal({"sku": sku, "scope": "organization"}, tool="provision", wrapped=False)
        before = call.model_dump()
        assert gate.review([call])[0].allowed is (sku in {"nimbus_e7", "nimbus_e8"})
        assert call.model_dump() == before  # No auto-map, no discarded override.


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"reason": "Customer specifically requested this record"},
        {"waive_early_closure_fee": True},
    ],
)
def test_does_not_forbid_legitimate_optional_changes_to_chase_reference(banking, extra):
    call = proposal({"account_id": "a7", **extra}, tool="close_bank_account_7392")
    before = call.model_dump()
    assert banking.public_argument_contracts.review([call])[0].allowed
    assert call.model_dump() == before
