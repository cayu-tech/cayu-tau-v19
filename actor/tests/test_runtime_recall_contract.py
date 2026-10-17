"""Characterize the pinned public runtime recall-source integration contract."""

import pytest
from cayu.memory.context import AutomaticRecallContextPolicy, AutomaticRecallSourceConfig
from test_runtime_banking import assembly


def test_native_recall_policy_rejects_silently_dropping_banking_fusion_channels(tmp_path):
    agent, _, captured = assembly(tmp_path, [])
    state = agent.get_init_state()
    try:
        corpus = captured["state"].memory.corpus
        with pytest.raises(ValueError, match="Fusion channels must exactly match"):
            AutomaticRecallContextPolicy(
                admission_policy=corpus.admission_policy,
                fusion_config=corpus.fusion_config,
                sources=AutomaticRecallSourceConfig(include_transcript=False),
            )
    finally:
        state.stop()
