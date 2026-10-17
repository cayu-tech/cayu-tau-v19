"""Candidate generic evidence-coverage guidance, evaluated before actor integration."""

EVIDENCE_SCOPE_GUIDANCE = """Before declaring a required check impossible, examine the documented coverage
of tools and results already available. Match the information needed, not the tool's name:
a complete inventory that includes inactive, closed or revoked records may already supply
historical evidence without a separate history endpoint. Use absence only within established
coverage: verify the relevant population, lifecycle states, time range and pagination. If a
complete applicable record set contains no disqualifying records, do not invent a missing
check solely because there is no separately named history tool. A current-only, filtered,
paginated or unspecified-scope result does NOT establish historical absence; finish available
pages or obtain the missing scope/evidence. Do not generalize one customer's records to others.
A prior plan blocker or summary is your own assessment, not authoritative policy or data.
Reconsider it against the actual tool contract and observations before abandoning authorized
work or offering an escalation. Do not weaken a real prerequisite or invent unavailable facts.
""".strip()
