---
name: verification-before-completion
description: Use before claiming that an implementation, bug fix, refactor, migration, configuration change, or review is complete. Verify using available tests, build output, static analysis, diff review, or clearly state what could not be verified.
argument-hint: "[change to verify]"
---

# Verification Before Completion

Do not declare success without current evidence.

## Process

1. Review the requested acceptance criteria.
2. Review all changed files.
3. Check for:
   - incomplete implementation,
   - unrelated edits,
   - dead code,
   - debug statements,
   - missing error handling,
   - security regressions,
   - missing tests,
   - documentation or configuration changes.
4. Run relevant verification when possible:
   - tests,
   - build,
   - lint,
   - type checking,
   - static analysis,
   - formatting.
5. Inspect actual command output.
6. Distinguish:
   - verified,
   - partially verified,
   - not verified.
7. Never say "all tests pass" unless the test output was observed.
8. Report remaining risks or manual steps.

## Output

### Verified

List evidence-supported results.

### Not verified

List unavailable checks or missing evidence.

### Remaining risks

List risks that still require attention.

### Completion status

Use one of:

- Complete and verified
- Implemented but partially verified
- Incomplete
