---
name: test-driven-development
description: Use when implementing behavior that can be expressed through automated tests. Define or write a failing test before implementing production code.
argument-hint: "[behavior to implement]"
---

# Test-Driven Development

Implement behavior through a red, green, refactor workflow.

## Process

1. Define the desired behavior precisely.
2. Identify the correct test level:
   - unit,
   - integration,
   - end-to-end.
3. Inspect existing test conventions.
4. Write or describe the smallest failing test.
5. Confirm that the test fails for the intended reason when execution is available.
6. Implement the minimum production change required to pass.
7. Run the relevant tests when execution is available.
8. Refactor without changing observable behavior.
9. Run the tests again.
10. Review edge cases and regression risks.

## Rules

- Do not weaken tests merely to make them pass.
- Do not mock the behavior being tested.
- Prefer behavior-oriented tests over implementation-detail tests.
- Do not claim that a test failed or passed without observing its output.

## Browser limitation

When tests cannot be executed:

- create or propose the test,
- explain why it should initially fail,
- clearly mark execution as pending.
