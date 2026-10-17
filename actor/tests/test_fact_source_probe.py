from dataclasses import replace

import pytest

from cayu_tau3_banking.fact_source_probe import CASES, STEPS, build_request


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_expectations_never_enter_actor_context(case):
    assert build_request(case, "prompt") == build_request(
        replace(case, expected="hidden"), "prompt"
    )
    request = build_request(case, "prompt")
    assert request.tools[0]["input_schema"]["properties"]["step"]["enum"] == list(STEPS)
