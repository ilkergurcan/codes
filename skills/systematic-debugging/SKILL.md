---
name: systematic-debugging
description: Use for defects, regressions, failing tests, runtime errors, intermittent failures, performance problems, and unexpected behavior. Gather evidence and determine the root cause before changing code.
argument-hint: "[bug, error, or failing behavior]"
---

# Systematic Debugging

Find the root cause before proposing or implementing a fix.

## Rules

Do not make speculative code changes before collecting evidence.

Do not claim a root cause unless it is supported by code, logs, test output, or reproducible behavior.

## Process

1. Define:
   - expected behavior,
   - actual behavior,
   - affected environment,
   - reproducibility.
2. Inspect the relevant execution path.
3. Gather available evidence:
   - error messages,
   - stack traces,
   - logs,
   - failing tests,
   - recent changes,
   - configuration,
   - data flow.
4. Form multiple hypotheses.
5. Rank hypotheses by likelihood and supporting evidence.
6. Design the smallest test or inspection that distinguishes between them.
7. Identify the root cause.
8. Propose the smallest safe fix.
9. Identify regression tests.
10. Verify the fix with available evidence.

## Browser limitation

When terminal execution is unavailable:

- do not pretend commands were run,
- inspect code statically,
- propose exact commands for a full environment,
- clearly separate verified findings from hypotheses.

## Output

Include:

- Symptom
- Evidence
- Hypotheses considered
- Root cause
- Proposed fix
- Regression test
- Verification status
