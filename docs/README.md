# `docs/`

- [`DESIGN.md`](DESIGN.md) — the authoritative design doc. All module
  READMEs link back to it by section. Change here first if you're
  rewriting architecture.

## Doc discipline

- The design doc is the spec. Per-module READMEs describe **how the
  implementation currently realizes it** and link to the relevant
  section.
- If implementation diverges from the design doc, open a PR against
  `DESIGN.md` first — don't let the drift accumulate silently in module
  READMEs.
- Prefer adding a new runbook under `docs/runbooks/` (create as needed)
  over growing this file.
