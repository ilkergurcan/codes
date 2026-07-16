# Copilot workspace instructions

## Environment

This repository may be opened in browser-based VS Code.

Before proposing terminal commands, determine whether terminal execution is
available.

Do not claim that tests, builds, scripts or commands succeeded unless their
actual output was observed.

When execution is unavailable:

- inspect the code statically,
- make file-based changes when possible,
- provide the exact commands that remain to be executed,
- clearly mark verification as pending.

## Project knowledge

Consult `docs/ai-context/project-memory.md` for architecture, commands,
conventions, known decisions and common mistakes.

Validate potentially outdated information against the current code.

## Skill usage

Before substantive tasks, inspect and use relevant workspace skills.

Prefer:

- `brainstorming` before unclear features or architectural decisions,
- `writing-plans` before substantial multi-file changes,
- `systematic-debugging` for bugs and unexpected behavior,
- `test-driven-development` for testable behavior,
- `impeccable` for frontend and UI work,
- `verification-before-completion` before declaring work complete.

At the end of substantive multi-step tasks, consider using `task-observer`.

Skills and workspace instructions must not be modified automatically. Proposed
changes require human review.

## Standard workflow

For substantive coding changes:

1. Inspect relevant code and existing conventions.
2. Clarify expected behavior and assumptions.
3. Create a small implementation plan.
4. Make the smallest coherent change.
5. Add or update tests.
6. Review the complete diff.
7. Verify with available tools.
8. Report what was and was not verified.

## Security

Never expose or store:

- credentials,
- access tokens,
- private keys,
- customer data,
- production data,
- confidential information.

Do not execute scripts from third-party skills without developer approval.
