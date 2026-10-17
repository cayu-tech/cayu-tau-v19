"""Bound policy correction work without interpreting customer intent."""


class AdmissionStallError(RuntimeError):
    """The actor exhausted its per-customer-turn rejected-round budget."""


class AdmissionProgress:
    """Count tool/response-rejected model rounds, not repeated policy evaluations.

    Only a new authoritative customer message resets the budget. Successful
    lookups and plan changes cannot replenish retries. Exhaustion is an error,
    never permission to bypass admission or report task completion.
    """

    def __init__(self, *, fatal: bool = True) -> None:
        self.fatal = fatal
        self.reset()

    def reset(self) -> None:
        self._rounds: set[str] = set()

    def observe_denial(self, model_step_id: str) -> None:
        # Saturation also bounds memory if a caller keeps reporting denials
        # after exhaustion. Runtime context checks enforce the actual stop.
        if len(self._rounds) < 8:
            self._rounds.add(model_step_id)

    @property
    def warning(self) -> str | None:
        if len(self._rounds) < 2:
            return None
        count = str(len(self._rounds)) if len(self._rounds) < 8 else "at least 8"
        warning = (
            f"Tool or response admission rejected {count} model rounds this customer turn. "
            "Do not repeat rejected operations without correcting the stated cause. "
            "Use authoritative evidence to repair the request, or explain a genuine blocker "
            "without claiming unexecuted work succeeded. "
        )
        if self.fatal:
            return warning + "Eight rejected rounds stop this attempt as an error. Plan edits and unrelated lookups do not reset this budget."
        return warning + "A corrected request or independent permitted work may continue within the task's overall limits. A warning grants no permission to bypass the rejected contract."

    def check(self) -> None:
        if self.fatal and len(self._rounds) >= 8:
            raise AdmissionStallError("admission_retry_budget_exhausted")
