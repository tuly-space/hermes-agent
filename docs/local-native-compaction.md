# Local native Responses compaction

The local integration supports GPT-5.6 and GPT-6 model families. GPT-6 includes `gpt-6`, `gpt-6-astra`, and dotted version variants; unrelated names such as `gpt-60` are excluded.

In addition to existing direct OpenAI support, the verified local Codex proxy route is allowed: `https://us-lrv03-sj4srl0f0-react-work.taila837d.ts.net/v1`. If provider identity is available it must be `custom:local-codex-proxy` (or normalized `custom`). This explicit endpoint allowlist is shared by capability resolution and request gating; arbitrary custom providers remain denied. Existing explicit runtime denial, checkpoint requirements, and compression-disable switches remain authoritative.

Live verification with `gpt-6-astra`: `/responses` plus `context_management=[{"type":"compaction","compact_threshold":1024}]` returned a compaction item; replaying only that encrypted item recalled all three fictional facts. The standalone `/responses/compact` route returned 404 and is not used.

Local source changes require a separately authorized Gateway restart before existing processes use them. The configured production threshold is unchanged.
