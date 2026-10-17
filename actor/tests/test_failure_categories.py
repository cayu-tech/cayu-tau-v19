import pytest

from cayu_tau3_banking.failure_categories import classify_exception, classify_failure_info
from cayu_tau3_banking.retrieval_eval import safe_error_diagnostic
from cayu_tau3_banking.runtime_agent import RuntimeCallbackExecutionError, RuntimeLimitError


@pytest.mark.parametrize(
    "name,expected",
    [
        ("RuntimeLimitError", "runtime_limit"),
        ("AdmissionStallError", "admission_stall"),
        ("PlanningStallError", "planning_stall"),
        ("RuntimeCallbackTimeoutError", "adapter_deadline"),
        ("ExecutionDeadlineExceeded", "execution_deadline"),
        ("ReadError", "provider_transport"),
        ("RuntimeShutdownError", "cleanup_failure"),
        ("TimeoutError", "unclassified"),
        ("AnythingUnknown", "unclassified"),
    ],
)
def test_typed_causes_remain_separate(name, expected):
    assert classify_failure_info({"error_type": name}) == expected
    assert classify_exception(RuntimeCallbackExecutionError(name)) == expected


def test_wrapper_is_exact_not_prose_guessing_or_sensitive_output():
    assert (
        classify_failure_info({"error_type": "RuntimeError", "error": "RuntimeLimitError"})
        == "unclassified"
    )
    for raw in (
        None,
        [],
        {"error_type": []},
        {
            "error_type": "RuntimeCallbackExecutionError",
            "error": "prefix Runtime callback failed (ReadError); conversation stopped.",
        },
    ):
        assert classify_failure_info(raw) == "unclassified"
    diagnostic = safe_error_diagnostic(ValueError("private customer or credential content"))
    assert "private" not in str(diagnostic)
    assert diagnostic["failure_category"] == "unclassified"
    assert safe_error_diagnostic(RuntimeLimitError())["failure_category"] == "runtime_limit"


def test_runtime_callback_error_retains_structured_failure_evidence():
    from cayu.failure_evidence import FailureEvidence

    evidence = FailureEvidence(classification="failure", exception_types=("ModelProviderError",))
    error = RuntimeCallbackExecutionError("ModelProviderError", evidence=evidence)
    assert error.failure_type == "ModelProviderError"
    assert error.failure_evidence is evidence


def test_structured_provider_failure_is_not_unclassified():
    assert (
        classify_failure_info(
            {
                "error_type": "RuntimeCallbackExecutionError",
                "error": "Runtime callback failed (ModelProviderError); conversation stopped.",
                "failure_evidence": {
                    "exception_types": ["ModelProviderError", "RemoteProtocolError"]
                },
            }
        )
        == "provider_transport"
    )
    assert classify_failure_info({"error_type": "ModelProviderError"}) == "provider_error"


def test_provider_diagnostic_quota_is_classified():
    assert classify_failure_info(
        {
            "error_type": "RuntimeCallbackExecutionError",
            "provider_error": {"error": {"type": "usage_limit_reached"}},
        }
    ) == "provider_quota"


@pytest.mark.parametrize("envelope,expected", [
    ({"error": {}, "http_status_code": 429}, "provider_rate_limit"),
    ({"error": {"code": "usage_limit_reached"}, "http_status_code": 429}, "provider_quota"),
    ({"error": {"type": "insufficient_quota"}, "http_status_code": 429}, "provider_quota"),
    ({"error": {"code": "server_is_overloaded"}}, "provider_unavailable"),
    ({"error": {}, "http_status_code": 503}, "provider_unavailable"),
    ({"error": {}, "http_status_code": 400}, "provider_error"),
])
def test_provider_http_status_fallback_preserves_explicit_cause(envelope, expected):
    assert classify_failure_info({"error_type": "ModelProviderError", "provider_error": envelope}) == expected
