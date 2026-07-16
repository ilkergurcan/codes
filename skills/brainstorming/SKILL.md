---
name: brainstorming
description: Use before implementing unclear features, changing product behavior, making architectural decisions, or when multiple valid implementation approaches exist. Clarify requirements and compare approaches before editing code.
argument-hint: "[feature, problem, or architectural decision]"
---

# Brainstorming

Understand and design the solution before implementation.

## Rules

Do not edit code until the problem and proposed design are sufficiently clear.

## Process

1. Inspect relevant repository files and existing patterns.
2. Restate the problem in concrete terms.
3. Identify:
   - users or systems affected,
   - desired behavior,
   - constraints,
   - edge cases,
   - security concerns,
   - backward compatibility requirements.
4. Separate confirmed facts from assumptions.
5. Present two or three realistic approaches when meaningful.
6. Compare approaches using:
   - complexity,
   - maintainability,
   - performance,
   - security,
   - testability,
   - compatibility with the existing architecture.
7. Recommend one approach and explain why.
8. Define acceptance criteria.
9. Stop before implementation unless the developer explicitly asks to continue.

## Output

Include:

- Problem definition
- Relevant repository findings
- Assumptions
- Options considered
- Recommended design
- Risks
- Acceptance criteria
