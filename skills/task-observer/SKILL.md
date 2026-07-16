---
name: task-observer
description: Use near the end of a substantive coding, debugging, planning, or review session to identify developer corrections, recurring workflow friction, missing repository instructions, and opportunities to improve Agent Skills.
argument-hint: "[task or session to review]"
---

# Task Observer

Review the current task for reusable lessons.

## Observe

Look for:

### Developer corrections

- Which assumptions were corrected?
- Which approaches were rejected?
- Which output or workflow did the developer prefer?

### Workflow friction

- Which information was repeatedly difficult to find?
- Which steps were unnecessarily repeated?
- Which commands, conventions or repository facts were missing?

### Skill gaps

- Was an existing skill incomplete?
- Did a skill activate at the wrong time?
- Was there no skill for a recurring workflow?

### Verification gaps

- Were completion claims made without evidence?
- Were browser limitations overlooked?
- Were unavailable terminal or runtime capabilities assumed?

## Output

### Observations

Only concrete observations supported by the current session.

### Proposed workspace-instruction changes

Suggest small changes to `.github/copilot-instructions.md`.

### Proposed skill improvements

For each proposal include:

- Target skill
- Observed problem
- Proposed change
- Expected benefit
- Possible risk

### New skill candidates

Only suggest a new skill when the workflow is likely to recur.

## Safety

Do not store or reproduce:

- secrets,
- credentials,
- customer information,
- personal information,
- production data,
- proprietary code excerpts,
- full chat transcripts.

Do not modify skills or workspace instructions automatically. Present proposals
for human review.
