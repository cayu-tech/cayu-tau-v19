import pytest

from cayu_tau3_banking.admission_progress import AdmissionProgress, AdmissionStallError


def test_counts_rounds_warns_before_exhaustion_and_resets():
    progress = AdmissionProgress()
    for _ in range(20):
        progress.observe_denial("same-round")
    assert progress.warning is None
    progress.check()
    for index in range(6):
        progress.observe_denial(str(index))
    assert progress.warning is not None
    progress.check()
    progress.observe_denial("last-round")
    with pytest.raises(AdmissionStallError, match="admission_retry_budget_exhausted"):
        progress.check()
    progress.reset()
    progress.check()
    assert progress.warning is None
