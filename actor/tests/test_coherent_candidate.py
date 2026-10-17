"""Mechanism tests use fictional entities; no reference answers in actor logic."""
import pytest
from test_case_conditions import execute, obligation, setup
from test_condition_readiness import effect_call
from test_savings_interest_audit import (
    _POLICY_DOSSIER_ID,
    _gold_interest_ledger,
    _task_095_arguments,
)

from cayu_tau3_banking.execution_receipts import ExecutionReceipts
from cayu_tau3_banking.product_type_binding import ProductTypeBindings
from cayu_tau3_banking.savings_interest_audit import invoke_savings_interest_audit


def test_exact_type_binding_does_not_select_unknown_or_ambiguous_products():
    records = [dict(category='business_savings', title='Fictional Saver Account: Terms', entry_id='public-a')]
    bindings = ProductTypeBindings(records)
    args = dict(account_class='Fictional Saver Account', account_type='savings')
    assert 'business_savings' in bindings.error('open_bank_account_4821', args)
    assert bindings.error('open_bank_account_4821', {**args, 'account_type': 'business_savings'}) is None
    assert bindings.error('open_bank_account_4821', {**args, 'account_class': 'Unknown'}) is None
    assert bindings.error('unrelated', args) is None
    bindings = ProductTypeBindings([*records, dict(category='personal_savings', title='Fictional Saver Account: Terms', entry_id='public-b')])
    assert bindings.error('open_bank_account_4821', args) is None


@pytest.mark.parametrize('failed', [False, True])
def test_receipt_settles_exact_step_after_its_own_invalidation(failed):
    call = effect_call()
    c, ledger, created = setup(obligations=[obligation(status='ready', planned_tool={'tool_name': call.tool_name, 'arguments': call.arguments})])
    assert not created.is_error
    receipts = ExecutionReceipts(ledger, c)
    c.before_condition_execution(call)
    execute(ledger, call.tool_call_id, call.tool_name, call.arguments, error=failed)
    c.after_condition_execution(call.tool_call_id)
    c.reconcile_conditions()
    receipts.observe_result(call.tool_call_id)
    step = c.export_state()['items'][0]['obligations'][0]
    assert step['status'] == ('ready' if failed else 'resolved')
    assert step.get('applied_to') == (None if failed else call.tool_call_id)
    if not failed:
        assert not c.review([call])[0].allowed


def extra(kind='independent_promotion'):
    return dict(name='Fictional promotion', apy_percent=0.2, component_kind=kind,
                benefit_id='benefit-one', source=dict(entry_id='policy', revision=1, quote='An independent promotion adds 0.2%.'))


def audit(components, validator=None):
    args = _task_095_arguments()
    args['other_stackable_components'] = components
    return invoke_savings_interest_audit(_gold_interest_ledger(), args, required_policy_dossier_id=_POLICY_DOSSIER_ID, component_source_validator=validator)


@pytest.mark.parametrize('kind', ['card_bonus', 'checking_boost'])
def test_relabeling_bonus_does_not_make_it_stackable(kind):
    result = audit([extra(kind)], lambda _: True)
    assert result.is_error and 'belong only' in result.content


def test_additional_components_need_source_and_cannot_duplicate_identity_or_clause():
    assert audit([extra()]).is_error
    assert audit([extra()], lambda _: False).is_error
    assert not audit([extra()], lambda _: True).is_error
    assert audit([extra(), {**extra(), 'name': 'Another label'}], lambda _: True).is_error
    assert audit([extra(), {**extra(), 'name': 'Another label', 'benefit_id': 'second'}], lambda _: True).is_error


def test_legacy_untyped_double_count_is_rejected():
    assert audit([dict(name='Card relationship benefit', apy_percent=0.025)]).is_error


def test_comparison_row_is_injected_after_compaction_without_extra_read(tmp_path):
    import json

    from cayu import ScriptedModelProvider
    from cayu.context.base import CheckpointCompactionContextPolicy, ModelCompactor
    from tau2.data_model.message import UserMessage
    from test_runtime_banking import assembly, call, text

    from cayu_tau3_banking.decision_notes import DECISION_NOTE_TOOL

    row = {
        'id': 'option-a', 'claim': 'Comparison awaiting the fee', 'state': 'unknown', 'sources': [],
        'comparison': {'alternative': 'Fictional Saver', 'calculation': '20000 * 0.05 - unknown fee',
                       'conditions': 'Customer does not want another checking account',
                       'next_missing_fact': 'Annual companion fee'}
    }
    summaries = ScriptedModelProvider([text('Earlier detail omitted.') for _ in range(20)])
    agent, provider, captured = assembly(
        tmp_path, [call(DECISION_NOTE_TOOL, {'facts': [row]}, 'note'),
                   text('What is your preference?'), text('The fee is still unknown.')],
        controller_mode='advisory', planning_mode='outcomes', advice_mode='lean', persistent=True,
        compaction=CheckpointCompactionContextPolicy(
            compactor=ModelCompactor(provider=summaries, model='scripted-summary'),
            compact_after_messages=2, max_user_turns=1),
    )
    state = agent.get_init_state()
    try:
        agent.generate_next_message(UserMessage.text('Compare the options.'), state)
        agent.generate_next_message(UserMessage.text('Keep the original scope.'), state)
        assert summaries.requests
        context = '\n'.join(p.text for m in provider.requests[-1].messages for p in m.content if p.type == 'text')
        assert json.dumps(row['comparison']) in context
        assert captured['state'].decision_facts.facts['option-a']['state'] == 'unknown'
    finally:
        state.stop()
