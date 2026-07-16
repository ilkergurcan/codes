---
name: writing-plans
description: Use after requirements are understood and before a substantial multi-file implementation, refactor, migration, or architectural change. Produce a concrete implementation plan with files, tests, risks, and verification steps.
argument-hint: "[feature or approved design]"
---

# Writing Plans

Create an implementation plan that another developer or coding agent can follow.

## Process

1. Inspect the relevant repository structure.
2. Identify existing patterns that should be followed.
3. Break the implementation into small, ordered steps.
4. For every step, specify:
   - files to create,
   - files to modify,
   - functions, classes, components, or modules involved,
   - intended behavior,
   - tests to add or update,
   - verification method.
5. Identify dependencies between steps.
6. Highlight:
   - migrations,
   - public API changes,
   - configuration changes,
   - security implications,
   - backward compatibility risks.
7. Avoid vague steps such as "implement the feature."
8. Do not modify code unless explicitly requested.

## Output format

### Goal

Describe the intended result.

### Relevant existing architecture

Summarize the repository patterns that affect the implementation.

### Implementation steps

Use ordered, concrete steps with file paths.

### Testing

List unit, integration, end-to-end, and manual tests where relevant.

### Risks

List implementation and rollout risks.

### Verification

State the commands or evidence required before completion.
