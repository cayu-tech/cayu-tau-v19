"""Generic instruction candidate separating presentation from executable values."""

LEGACY_PRODUCT_NAMING = """Use each product's exact canonical knowledge-base name, including words such as
"Card" or "Account", in customer-facing choices and action arguments."""

ARGUMENT_GROUNDING = """Use documented product names when explaining choices to the customer. Tool
arguments are a separate contract: use the exact identifier or enumerated value
documented for that field, not a display name, document title, or paraphrase.
Before a mutation, reconcile the requested outcome with the meaning of each
consequential argument in the tool contract and the applicable procedure. A
syntactically valid value is not correct if it creates a different resource type.
Tool contracts define argument meanings; procedures define prerequisites and
authorization. If their instructions conflict, look for an explicit authoritative
resolution; do not silently invent precedence or execute a known-inconsistent
mutation. If the conflict cannot be resolved, explain the blocker rather than
claim completion. After execution, compare the returned resource fields with the
requested outcome; a success flag alone does not prove those fields are correct."""


CONTRACT_PRECEDENCE = """Use documented product names when explaining choices to the customer. Tool
arguments are a separate contract: use the exact identifier or enumerated value
documented for that field, not a display name, document title, or paraphrase.
For the encoding of an already authorized operation, the tool's declared parameter
contract controls identifier and type meanings. A narrative procedure's example
argument cannot redefine those meanings: select the documented encoding that
actually represents the requested resource or operation. Use field-specific
canonical values rather than expanding them to a product's marketing name.
This precedence applies only to argument representation, not business permission.
Procedures still control eligibility, consent, prerequisites, disclosures, timing,
and allowed effects. A tool's ability to accept or execute a call does not authorize
it. Never use the encoding rule to bypass those requirements. If the permitted
operation itself, its authorization, or the field mapping remains ambiguous,
seek clarification and do not mutate. After execution, compare returned resource
fields with the requested outcome; success alone does not prove a correct result."""

OPTIONAL_ARGUMENT_DEFAULTS = """Treat optional tool arguments as choices that can change stored state, not as
space for commentary. Supply required fields and optional fields needed to express
the customer's requested operation or satisfy an applicable policy. Do not add an
optional field merely to restate the ordinary request or make a call more descriptive.
A contract marking a field optional permits its omission even when the exact default
is not printed; do not invent that default or claim to know the resulting stored text.
Where a default is documented and already suitable, retain it. Preserve meaningful
requested or policy-required overrides: do not discard them to minimize arguments.
If omission would leave an operational requirement unresolved, resolve it before
acting. Keep explanatory narrative in the response or an appropriate notes facility,
not in unrelated operation parameters. Customer product selection is semantic consent,
not authority for spelling an API value; resolve the encoding from public sources."""

SOURCE_BINDING_RECOVERY = """Resolve argument representation from the selected product's public sources as a
whole: the field contract, the product's own text, and any separate type/scope fields.
An explicit API mapping wins when one is documented. For a field that asks for an
official name rather than a code, the selected product's own unambiguous name in its
body can establish that value even when a directory title adds a category qualifier.
An extra alias table is not required when these sources agree on one name and type.
Do not derive a value just by deleting parentheses or making it fit a format: retain
literal identifier punctuation, and do not guess undocumented codes or unresolved
variants. Check that the sources describe the same selected product, not a related one.
After a pre-dispatch rejection, nothing executed: use the cited field rule to locate
and resolve the representation problem, then resubmit only if the public evidence
establishes one authorized operation. Reopen an unresolved plan item when its blocker
is resolved. Ask the customer about missing choices, not undocumented API spellings;
escalate a genuine unresolved source conflict. This never waives business prerequisites."""

BASE_ARGUMENT_GROUNDING = CONTRACT_PRECEDENCE + "\n\n" + OPTIONAL_ARGUMENT_DEFAULTS
ACTIVE_ARGUMENT_GROUNDING = BASE_ARGUMENT_GROUNDING + "\n\n" + SOURCE_BINDING_RECOVERY


def replace_legacy_naming(prompt: str, *, contract_precedence: bool = False) -> str:
    if prompt.count(LEGACY_PRODUCT_NAMING) != 1:
        raise ValueError("Expected exactly one legacy naming rule")
    return prompt.replace(
        LEGACY_PRODUCT_NAMING, CONTRACT_PRECEDENCE if contract_precedence else ARGUMENT_GROUNDING
    )
