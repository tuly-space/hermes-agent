# Local native Responses compaction

The local integration preserves upstream's GPT-5.6 route gates and exact `gpt-6-astra` support on official Codex OAuth. On the explicitly verified local proxy below, it additionally supports the GPT-6 family (`gpt-6`, `gpt-6-astra`, and dotted version variants); unrelated names such as `gpt-60` are excluded. This exception does not enable GPT-6 on arbitrary custom or direct API routes.

In addition to existing direct OpenAI support, the verified local Codex proxy route is allowed: `https://us-lrv03-sj4srl0f0-react-work.taila837d.ts.net/v1`. If provider identity is available it must be `custom:local-codex-proxy` (or normalized `custom`). This explicit endpoint allowlist is shared by capability resolution and request gating; arbitrary custom providers remain denied. Existing explicit runtime denial, checkpoint requirements, and compression-disable switches remain authoritative.

Live verification with `gpt-6-astra`: `/responses` plus `context_management=[{"type":"compaction","compact_threshold":1024}]` returned a compaction item; replaying only that encrypted item recalled all three fictional facts. The standalone `/responses/compact` route returned 404 and is not used.

Local source changes require a separately authorized Gateway restart before existing processes use them. The configured production threshold is unchanged.

## Native maintenance and next-turn idle compaction

`compression.idle_compact_after_seconds` uses the existing turn-start gap trigger; it does not schedule background work. The existing idle size, cooldown, and lock guards determine whether a pass runs. Hermes emits its existing idle compression status when a pass is admitted; the gateway delivers this idle-resume status without enabling other routine compression progress (`compression.progress_notices` remains their gate).

`auxiliary.compression.native: true` permits the supported Sol auxiliary route to attempt an inline `/responses` checkpoint when the existing compression flow admits a pass, including idle resume. `auxiliary.compression.preserve_reasoning: true` retains eligible encrypted reasoning on that route. A completed response without a compaction item falls back to the ordinary summary. Raw transcript rows remain intact and checkpoints are persisted on assistant rows. The auxiliary-native request uses the effective `auxiliary.compression.timeout`; this does not change the total compression ceiling or prove provider-side deadlines.
