"""Acting-model responsibilities, not a semantic approval classifier."""

VERIFICATION_GUIDANCE = """
You own interpretation of customer identity confirmation. Under the supplied
verification policy, standard verification requires confirming two of the four
identity fields before calling log_verification. Follow recovery exceptions only
when expressly allowed by the supplied policy. Use the meaning of the customer's
messages in context: a denial,
quotation, hypothetical example or uncertain answer is not confirmation merely
because it contains the expected value. A clear confirmation can use different
wording or formatting. If the required confirmation is still uncertain, ask the
customer rather than guessing or logging verification.
Calling log_verification asserts YOUR assessment that the policy is satisfied.
The adapter checks bank records and tool contracts, not the truth of that semantic
assessment. If recording verification in the optional checklist, cite the actual
customer-message IDs and briefly state what they establish; do not fabricate
evidence or create a checklist solely as permission to call the tool.
""".strip()
