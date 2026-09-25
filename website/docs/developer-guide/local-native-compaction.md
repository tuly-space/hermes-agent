# Local native Responses compaction

The local integration preserves upstream's GPT-5.6 route gates and exact `gpt-6-astra` support on official Codex OAuth. On the explicitly verified local proxy below, it additionally supports the GPT-6 family (`gpt-6`, `gpt-6-astra`, and dotted version variants); unrelated names such as `gpt-60` are excluded. This exception does not enable GPT-6 on arbitrary custom or direct API routes.

In addition to existing direct OpenAI support, the verified local Codex proxy route is allowed: `https://us-lrv03-sj4srl0f0-react-work.taila837d.ts.net/v1`. If provider identity is available it must be `custom:local-codex-proxy` (or normalized `custom`). This explicit endpoint allowlist is shared by capability resolution and request gating; arbitrary custom providers remain denied. Existing explicit runtime denial, checkpoint requirements, and compression-disable switches remain authoritative.

Live verification with `gpt-6-astra`: `/responses` plus `context_management=[{"type":"compaction","compact_threshold":1024}]` returned a compaction item; replaying only that encrypted item recalled all three fictional facts. The standalone `/responses/compact` route returned 404 and is not used.

Local source changes require a separately authorized Gateway restart before existing processes use them. The configured production threshold is unchanged.

## Optional native-first and background idle pilot

For a profile that has already enabled `compression.codex_responses_native: true`, set
`compression.codex_responses_native_first: true` to attempt native `/responses`
maintenance before automatic local summary compression. Set
`compression.codex_responses_native_idle_after_seconds: 1500` for a one-shot
background check 25 minutes after each completed durable turn. The profile-owned
idle timetable is an atomic data file: this update backfills currently bound,
completed idle sessions once; later restarts load the file, preserve original
deadlines, and never rescan history. A single condition-driven scheduler wakes at
the next deadline or on a timetable/worker/shutdown event and runs at most two
compression workers, without per-session gateway timers or a polling backlog.
Non-gateway sessions retain their existing local timer. Both new
settings default to off; keep the older `idle_compact_after_seconds: 0` or that separate
next-turn summary mechanism may also run. Idle maintenance requires ≥80K
effective full-request input tokens (configurable with
`compression.codex_responses_native_idle_min_tokens`) and ≥16K growth since
its previous checkpoint/attempt. This floor does not guarantee savings.
The attempt watermark is durable and the checkpoint is attached to the existing
assistant row, never replacing searchable raw transcript rows or changing session ID.
No completed compaction item means failure. Idle never sends its model output as
chat. A due check for a parent session with a running/finalizing delegation or an
undelivered (pending) completion is retired without a provider request, start
notice, or cached-agent eviction. This read-only gate consults that profile's
durable `state.db` ledger by the parent session ID, not the reusable gateway
routing key or process-local registry; a ledger read error also skips the pass.
Delivered or terminally dropped completions no longer block it. There is no
retry/polling for a skipped deadline: processing a completion as a normal parent
turn re-arms a full 25-minute delay after that turn, independently of when the
completion delivery is acknowledged. New gateway turns invalidate the saved
deadline and abort an in-flight idle request; the network wait holds no
session-turn lease. A short atomic commit checks that no foreground turn owns
the lease and the message watermark is unchanged.

For a Discord session originating in a thread, an eligible background pass
sends one brief start notice to that same thread when native `/responses`
maintenance begins (or when ordinary summary fallback actually starts if the
native request could not begin). This is independent of routine
`compression.progress_notices`. Skipped, below-floor, stale or pre-start
cancelled timers stay silent. A failed native attempt followed by summary
fallback does not send a second start notice. Once a native checkpoint or
ordinary fallback is durably committed, the same original Discord thread also
receives “上下文压缩已完成，下次对话会接续压缩结果。” Failed persistence, cancellation,
and skipped checks do not receive a completion notice. Neither notice creates a
session message; delivery failure does not change the committed result.
Channel/DM sessions and other platforms receive no notice.

Maintenance logs distinguish full-request *rough before* pressure from actual
maintenance request `usage_input/usage_cached/usage_output`; after-checkpoint
input remains unknown until an ordinary provider response reports it. Opaque
ciphertext length is not token usage. Every new foreground turn invalidates its
previous idle plan. Its completed turn updates the persistent idle time, while an
idle worker conditionally removes
only the version it claimed, even after a failure or below-threshold check. It
will not retry that session until another completed conversation turn. The worker
keeps raw history and the logical session intact and releases its own resources.
Malformed timetable files fail closed rather than triggering a new history scan.
A cross-process replay proves persistence, not an undocumented checkpoint
lifetime guarantee.
