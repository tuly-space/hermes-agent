"""Real preflight/facade/SQLite regressions; only model transport is mocked."""
import copy
import itertools
import logging
from types import SimpleNamespace

import pytest

from tests.agent.test_unified_auto_compaction import agent, _native_fixture


@pytest.mark.parametrize('idle,tokens,count', [x for x in itertools.product([False, True], repeat=3) if any(x)])
@pytest.mark.parametrize('outcome', ['checkpoint', 'missing', 'cancelled', 'stale'])
def test_normalized_checkpoint_branch_matrix(agent, monkeypatch, caplog, idle, tokens, count, outcome):
    from agent import conversation_compression, title_generator
    from agent.native_compaction import resolve_native_compaction_capabilities
    instance, db = agent
    instance.api_mode = 'codex_responses'
    instance.provider = 'openai-codex'
    instance.base_url = 'https://chatgpt.com/backend-api/codex'
    instance.model = 'gpt-6-astra'
    instance.compression_aux_native = True
    instance.compression_aux_provider = 'openai-codex'
    instance.compression_aux_model = 'gpt-6-sol'
    instance.compression_aux_base_url = None
    instance.runtime_capabilities = resolve_native_compaction_capabilities(
        model=instance.model, provider=instance.provider, base_url=instance.base_url, is_codex_backend=True)
    instance.compression_idle_compact_after_seconds = 1 if idle else 0
    instance.compression_hard_message_limit = 5 if count else 5000
    instance.context_compressor.threshold_tokens = 100 if tokens else 50000
    monkeypatch.setattr('agent.turn_context._should_idle_compact', lambda **kw: idle)
    # Deliberately preserve trailing whitespace in both original and durable history.
    history = [{'role': 'user' if i % 2 == 0 else 'assistant',
                'content': ('long ' * 120) if tokens else f'{i} ', 'timestamp': 1} for i in range(6)]
    db.create_session('unified', 'cli', model=instance.model)
    db.append_messages_batch('unified', copy.deepcopy(history))
    monkeypatch.setattr(title_generator, '_auto_title_enabled', lambda: False)
    if outcome == 'stale':
        history[0]['content'] += ' memory diverged from durable input'
    calls, summaries = [], []
    def create(**kwargs):
        calls.append({**kwargs, **kwargs.get('extra_body', {})})
        if kwargs['model'] == 'gpt-6-sol':
            if outcome == 'cancelled':
                instance.hard_interrupt()
            events = [] if outcome == 'missing' else [{'type': 'response.output_item.done',
                'item': {'type': 'compaction', 'encrypted_content': 'mock-checkpoint'}}]
            return iter(events + [{'type': 'response.completed', 'response': {'status': 'completed', 'output': []}}])
        return iter([{'type': 'response.output_text.delta', 'delta': 'done'},
            {'type': 'response.completed', 'response': {'status': 'completed', 'output': [],
             'usage': {'input_tokens': 42, 'output_tokens': 2}}}])
    instance._create_request_openai_client = lambda **kw: SimpleNamespace(responses=SimpleNamespace(create=create))
    instance._close_request_openai_client = lambda *a, **kw: None
    instance._abort_request_openai_client = lambda *a, **kw: None
    def summary(_agent, messages, system, **kw):
        summaries.append(1)
        instance._last_compaction_in_place = True
        instance.context_compressor.threshold_tokens = 50000
        return [{'role': 'assistant', 'content': 'mock summary'}, messages[-1]], system
    monkeypatch.setattr(conversation_compression, 'compress_context', summary)
    with caplog.at_level(logging.INFO):
        result = instance.run_conversation('next', conversation_history=history)
    expected_reasons = ','.join(n for n, enabled in [('idle', idle), ('tokens', tokens), ('message_count', count)] if enabled)
    assert 'reasons=' + expected_reasons in caplog.text
    assert sum(c['model'] == 'gpt-6-sol' for c in calls) == (0 if outcome == 'stale' else 1)
    assert len(summaries) == (1 if outcome == 'missing' else 0)
    if outcome in {'checkpoint', 'missing'}:
        assert result['final_response'] == 'done'
        assert sum(c['model'] == 'gpt-6-astra' for c in calls) == 1
        if outcome == 'checkpoint':
            assert any(i.get('type') == 'compaction' for i in calls[-1]['input'])
    else:
        assert result.get('compression_deferred') or result.get('interrupted')
        assert not any(c['model'] == 'gpt-6-astra' for c in calls)
    if outcome == 'stale':
        assert result['compression_deferred_reason'] == 'transcript_content_mismatch'
        assert 'unknown guard' not in caplog.text


@pytest.mark.parametrize('shape', ['whitespace', 'tool_metadata', 'api_content'])
def test_checkpoint_cold_replay_and_changed_prefix_rejection(agent, monkeypatch, shape):
    from agent import native_maintenance
    from agent.conversation_compression import CompressionCommitFence
    from agent.codex_responses_adapter import has_replayable_native_compaction_checkpoint
    from agent.turn_request_assembly import maintenance_api_prefix
    instance, db, _ = _native_fixture(agent, monkeypatch)
    history = db.get_messages_as_conversation('unified', include_row_ids=True)
    if shape == 'whitespace':
        history += [{'role': 'user', 'content': ' new fact \n'}, {'role': 'assistant', 'content': ' noted \n'}]
    elif shape == 'api_content':
        history += [{'role': 'user', 'content': 'display', 'api_content': 'effective user context'},
                    {'role': 'assistant', 'content': 'noted'}]
    else:
        history += [{'role': 'user', 'content': 'run tool'},
            {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_1', 'call_id': 'call_1',
            'response_item_id': 'fc_1', 'type': 'function', 'function': {'name': 'terminal', 'arguments': '{"x":1}'}}]},
            {'role': 'tool', 'tool_call_id': 'call_1', 'content': ' result \n'}]
    assert instance._flush_messages_to_session_db(history)
    before = copy.deepcopy(history)
    watermark = db.get_active_message_watermark('unified')
    instance.client.responses.create.side_effect = lambda **kw: iter([
        {'type': 'response.output_item.done', 'item': {'type': 'compaction', 'encrypted_content': 'mock-checkpoint'}},
        {'type': 'response.completed', 'response': {'status': 'completed', 'output': []}},
    ])
    assert native_maintenance.attempt(instance, history, 'Stable prompt', 1000, phase='threshold',
                                      expected_watermark=watermark, commit_fence=CompressionCommitFence())
    assert [(m['role'], m.get('content'), m.get('tool_calls')) for m in history] == [
        (m['role'], m.get('content'), m.get('tool_calls')) for m in before]
    # Reopen SQLite, not only the in-memory sidecar. No new model request needed.
    from hermes_state import SessionDB
    cold = SessionDB(db_path=db.db_path)
    try:
        restored = cold.get_messages_as_conversation('unified', repair_alternation=True, include_row_ids=True)
    finally:
        cold.close()
    view = maintenance_api_prefix(instance, restored, 'Stable prompt')
    assert has_replayable_native_compaction_checkpoint(instance, view)
    altered = copy.deepcopy(restored)
    altered[0]['content'] = 'different historical fact'
    assert not has_replayable_native_compaction_checkpoint(instance, maintenance_api_prefix(instance, altered, 'Stable prompt'))


@pytest.mark.parametrize('reason,attribute,value,expected', [
    ('native_abort', '_native_maintenance_abort_reason', 'watermark_mismatch', 'watermark_mismatch'),
    ('native_abort', '_native_maintenance_abort_reason', None, 'native_abort'),
    ('message_limit', '_compression_blocked_transient', None, 'message_limit'),
    ('transient_block', '_compression_blocked_transient', 'cooldown:42', 'cooldown:42'),
    ('lock', '_compression_skipped_due_to_lock', 'other', 'lock'),
])
def test_deferred_reason_contract(reason, attribute, value, expected):
    from agent.conversation_loop import _compression_deferred_result
    instance = SimpleNamespace(session_id='reason-test', **{attribute: value})
    result = _compression_deferred_result(instance, [], 0, reason=reason)
    assert result['compression_deferred_reason'] == expected
    assert result['compression_deferred'] and not result['failed']
    if reason in {'native_abort', 'message_limit'}:
        assert 'recent failed attempt' not in result['final_response']
        assert 'already running' not in result['final_response']
