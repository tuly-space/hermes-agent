# agent/ — AIAgent, turn loop, prompt, compression

Applies on top of the root `AGENTS.md` (prompt-caching invariant, facade + siblings rules).

## Shape

`run_agent.py` is the public facade: `AIAgent` is assembled from mixins (`agent/turn_facade.py`,
`client_lifecycle.py`, `stream_delivery.py`, `session_persistence.py`, `compression_facade.py`, ...).
Construction runs `agent/agent_init.py::init_agent`; a turn is
`agent/conversation_loop.py::run_conversation`, which `AIAgent.run_conversation` forwards to after
taking the session turn lease (`turn_facade_lease.py`). `AIAgent.__init__` takes ~60 parameters
(credentials, routing, callbacks, session context, budget, credential pool, ...) — read
`run_agent.py` for the list; the subset you usually touch: `base_url`, `api_key`, `provider`,
`api_mode` (`"chat_completions" | "codex_responses" | ...`), `model` (empty → resolved from
config/provider later), `max_iterations` (default 500, shared with subagents),
`enabled_toolsets`/`disabled_toolsets`, `quiet_mode`, `save_trajectories`, `platform`
(`"cli"`, `"telegram"`, ...), `session_id`, `skip_context_files`, `skip_memory`, `credential_pool`.
`chat(message) -> str` is the simple interface; `run_conversation(user_message, system_message=None,
conversation_history=None, task_id=None) -> dict` returns `final_response` + `messages`.

## Agent loop (`agent/conversation_loop.py` + `agent/turn_*.py`)

Entirely synchronous, with interrupt checks, budget tracking, and a one-turn grace call:

```python
while (api_call_count < self.max_iterations and self.iteration_budget.remaining > 0) \
        or self._budget_grace_call:
    if self._interrupt_requested: break
    response = client.chat.completions.create(model=model, messages=messages, tools=tool_schemas)
    if response.tool_calls:
        for tc in response.tool_calls:
            messages.append(tool_result_message(handle_function_call(tc.name, tc.args, task_id)))
        api_call_count += 1
    else:
        return response.content
```

Each phase of an iteration is its own sibling, so a change to (say) overflow handling touches one
~600-line file: `turn_preflight*`, `turn_iteration_prep`, `turn_request_assembly`/`turn_api_request`,
`turn_api_call`, `turn_api_error`, `turn_response_intake`/`turn_response_check`,
`turn_empty_response`, `turn_tool_round`/`turn_tool_validation`, `turn_overflow`,
`turn_truncation`, `turn_context_compaction`, `turn_recovery`, `turn_recovery_autorecover`
(post-exhaustion wait-and-retry ladder), `turn_retry_state`,
`turn_stop_gates`, `turn_liveness`, `turn_usage`, `turn_final_response`, `turn_finalizer`,
`turn_summary`. Find the phase with `grep -rn "def X" agent/turn_*.py`.

Messages use OpenAI format `{"role": "system|user|assistant|tool", ...}`; reasoning content is stored
in `assistant_msg["reasoning"]`.

**Agent-level tools** (`todo`, `memory`, ...) are intercepted by `agent/tool_executor.py` through the
`INLINE_TOOL_EXECUTORS` table in `agent/inline_tool_executors.py` before `handle_function_call()`.
Adding one: register in that table (no `if name == ...` chain); `tools/todo_tool.py` is the pattern.

## Message-flow invariants (every change is reviewed against these)

- **Prompt caching must not break.** Never alter past context, change toolsets, reload memories,
  or rebuild the system prompt mid-conversation. The system prompt is byte-stable for the life of
  a conversation; the ONLY context mutation is compression. Anything that must inject content
  mid-conversation rides a **user message or tool result**, never the system prompt: skill slash
  commands (`agent/skill_commands.py`) inject as a user message; subdirectory `AGENTS.md` hints
  (`agent/subdirectory_hints.py`) append to the tool result (head+tail truncated past `_MAX_HINT_CHARS = 32_000`;
  the truncation is logged, never queued as a chat status warning — `context_file_max_chars` does not raise that cap).
- **Strict role alternation.** Never two same-role messages in a row; never a synthetic user
  message injected mid-loop. The one exception is `/steer`, delivered as a standalone user row
  after a tool result (`assistant(tool_calls) → tool → user` is legal on every provider path) —
  never smeared onto the already-persisted tool row, which append-only persistence would leave
  divergent from the live request. Cron deliveries live in their own session for this reason.
- **Context files** (`agent/prompt_builder.py`) load from the CWD only at startup and are capped
  (`CONTEXT_FILE_MAX_CHARS` / dynamic cap from the context window / `context_file_max_chars`).
  Never load an install-tree `AGENTS.md` as project context (PR #64611); subdirectory hints reject
  paths outside the working dir so `~/.codex/AGENTS.md` / `~/.claude/CLAUDE.md` never mix in.
- **`_last_resolved_tool_names` is a process-global in `model_tools.py`.** `_run_single_child()`
  in `tools/delegate_tool.py` saves/restores it around subagent execution; code reading it may see
  a temporarily stale value during child runs.

## Compression (`agent/compression_facade.py`, `conversation_compression.py`, `turn_context_compaction.py`)

Manual `/compress` on every surface (CLI, gateway, TUI, ACP) runs through
`agent/conversation_compression_manual.py::compress_now` (one parser for `here [N]` / focus /
`--preview` / `--aggressive`; surfaces only parse their own argv, install `after_messages` and render).

Automatic compaction has one decision point at the assembled pre-API request gate
(`agent/turn_preflight.py`): combine resumed-session idle, token pressure, effective request message
count and engine maintenance into one `_compress_context` chain. Gateway, turn-start and post-tool
paths must not start competing automatic compression; tool-result growth is checked before the next
request. Cold sessions are considered on resume, not by a background idle timer. Manual `/compress`
and provider-proven overflow retain their separate recovery entry points.

Eligible native Responses/Codex maintenance runs first; ordinary failure falls back to the auxiliary
summary. Native checkpoint success waits for real usage; ordinary summaries may be remeasured and
retried within the existing budget, but waiting for usage must not trigger another idle summary.
Preserve cooldown, locks, cancellation, commit fences and review-fork guards. Native waits are bounded
by the effective auxiliary timeout policy; cancellation or stale-input/CAS refusal must not publish
late results, summarize old input or send an outdated request. Emit the existing scenario-appropriate
status once per combined decision, without another start notice on fallback.

In-place compaction keeps a stable session id. A stalled summary stream retries once on
`auxiliary.compression.fallback_chain`; repeated stalls use the deterministic fallback through the
same lease/fence pipeline. Compression is the sanctioned cache break — keep it the only one.
See `website/docs/developer-guide/local-native-compaction.md` for the local integration and
`website/docs/developer-guide/context-compression-and-caching.md` for the underlying pipeline.

## Model and provider resolution

- Runtime provider/model resolution and its precedence: `website/docs/developer-guide/provider-runtime.md`.
  Provider profiles are plugins (`plugins/model-providers/<name>/`, see `plugins/AGENTS.md`);
  `agent/model_metadata.py` holds context lengths and capabilities.
- **Auxiliary (side-LLM) work** — curator, vision, embedding, title generation, session_search,
  compression — resolves through `agent/auxiliary_client.py::_resolve_auto_route`; each task can pin
  its own `provider/model/base_url/reasoning_effort` under `auxiliary:` in config.yaml.
  Every physical attempt funnels through `_relay_sync_completion` / `_relay_async_completion` /
  `_relay_sync_stream`, where `agent/auxiliary_hooks.py` emits `pre_auxiliary_call` /
  `post_auxiliary_call` (observer-only, fail-open, `aux_task` set); the main-loop
  `pre/post_api_request` events must NOT fire for aux calls (#79733).
- Fallback models and credential pools are resolution-chain code: E2E them with real imports
  against a temp `HERMES_HOME`, not mocks (root rubric).

## Memory, context engines, curator

`agent/memory_provider.py` (ABC) + `agent/memory_manager.py` (orchestrator) drive memory-provider
plugins; `agent/context_engine.py` drives context-engine plugins; `agent/image_gen_provider.py`
image-gen plugins (all in `plugins/AGENTS.md`). `agent/curator.py` + `curator_backup.py` implement
the skill curator (`skills/AGENTS.md`). Cron sessions pass `skip_memory=True` by default — memory
providers intentionally do not run during cron.

- End-of-session memory extraction and provider `on_session_end` run wherever the session ends —
  turn, eviction, shutdown, `tui_gateway` teardown — and the CALLER binds the owning profile's scope
  first (`_run_release_in_profile_scope`, `_session_profile_runtime_scope`); the agent never derives
  its home from `os.environ` at flush time (`Path(_session_db.db_path).parent` is the ground truth).
  Provider background work starts through `memory_provider.py::spawn_context_thread` (copies the
  contextvars), never a bare `threading.Thread`; `title_generator.py` is the shape.
- `agent/secret_scope.py::get_secret` fails closed (`UnscopedSecretError`) only after
  `set_multiplex_active(True)`; the gateway, cron, migrate and `serve` set it. A new multi-home host
  must too, or every guard is silently off. Isolation is BETWEEN profiles; children inherit via
  `copy_context`; a child's `UnscopedSecretError` is a spawn-site bug, never grounds for an
  `os.getenv` fallthrough. Delegated children carry `delegation_context.py::
  DELEGATED_CHILD_ENV_MARKER` valued as the fenced Kanban board root, not a bare flag.

## Tests

Loop/phase tests go in `tests/agent/`; patch the binding the phase actually reads (siblings often
`from run_agent import X` inside the function — root "patch where production reads"). Assert
message-shape invariants (alternation, byte-stable system prompt) rather than snapshotting prompt
text.

Long-form: `website/docs/developer-guide/agent-loop.md`, `prompt-assembly.md`,
`context-compression-and-caching.md`, `provider-runtime.md`, `session-storage.md`,
`subagent-lifecycle-api.md`.
