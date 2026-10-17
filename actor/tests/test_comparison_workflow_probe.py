from decimal import Decimal

from cayu_tau3_banking import agent
from cayu_tau3_banking.comparison_workflow_probe import RETENTION_GUIDANCE, catalogue_case


def test_experimental_guidance_is_not_promoted_and_respects_memory_switch(monkeypatch):
    guidance = RETENTION_GUIDANCE
    assert guidance not in agent.banking_system_prompt("Fictional policy", planning_mode="outcomes")
    monkeypatch.setattr(
        agent, "_MEMORY_INSTRUCTIONS", agent._MEMORY_INSTRUCTIONS + "\n\n" + guidance
    )
    assert guidance in agent.banking_system_prompt("Fictional policy", planning_mode="outcomes")
    assert guidance not in agent.banking_system_prompt("Fictional policy", memory_enabled=False)
    assert "task_065" not in guidance
    assert "Simple one-step answers still need no worksheet" in guidance


def test_catalogue_probe_has_unique_sources_and_a_real_winner_change():
    documents, case, expected = catalogue_case()
    assert len(documents) == len({row[0] for row in documents}) == 16
    terms = {
        "Alder": ("4", "0"),
        "Birch": ("5", "30"),
        "Cedar": ("3.2", "0"),
        "Dune": ("4.5", "14"),
    }
    winners = []
    for balance in (2000, 4000):
        nets = {
            name: Decimal(balance) * Decimal(rate) / 100 - Decimal(fee) - 20
            for name, (rate, fee) in terms.items()
        }
        winners.append(max(nets, key=nets.get))
    assert winners == ["Alder", "Birch"]
    assert expected == ["Alder and Fern: $60", "Birch and Fern: $150"]
    assert "$60" not in case["customer"]
    assert "$150" not in case["followups"][0]
