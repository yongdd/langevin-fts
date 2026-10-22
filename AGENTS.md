# AGENTS.md

Read [CLAUDE.md](CLAUDE.md) in full before working in this repository. It contains
the shared workflow rules, build commands, validation requirements and architecture
guidance for all coding agents.

## Codex Notes

- Follow the active environment's sandbox and approval policy. GPU access errors
  do not imply that permission escalation is available or required.
- Use the scheduler's allocated GPU devices on shared systems. Follow the local
  resource policy rather than assuming a particular GPU index or cluster setup.
- State material assumptions, consider numerical edge cases and check conclusions
  against the current source and relevant tests.

Keep these project guides in English and independent of personal accounts,
machine-specific paths, credentials and private infrastructure details.
