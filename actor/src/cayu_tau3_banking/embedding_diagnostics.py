"""Allowlisted embedding failure metadata; never serialize provider messages/bodies."""

_CODES = frozenset(
    {
        "invalid_api_key",
        "insufficient_quota",
        "rate_limit_exceeded",
        "model_not_found",
        "invalid_request_error",
        "server_error",
    }
)


def embedding_failure_details(error: BaseException) -> dict:
    chain = []
    seen = set()
    while error is not None and id(error) not in seen and len(chain) < 6:
        seen.add(id(error))
        row = {"error_type": type(error).__name__}
        status = getattr(error, "status_code", None)
        if type(status) is int and 100 <= status <= 599:
            row["http_status"] = status
        for field in ("error_code", "code"):
            code = getattr(error, field, None)
            if isinstance(code, str) and code in _CODES:
                row["provider_code"] = code
        chain.append(row)
        error = error.__cause__ or (None if error.__suppress_context__ else error.__context__)
    return {"stage": "embedding_provider", "exception_chain": chain}


class EmbeddingInitializationError(RuntimeError):
    def __init__(self, diagnostic: dict):
        super().__init__("Embedding initialization failed; sanitized provider details recorded.")
        self.diagnostic = diagnostic
