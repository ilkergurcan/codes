---
name: impeccable
description: Use when designing, implementing, reviewing, auditing, critiquing, or polishing frontend user interfaces. Apply visual hierarchy, typography, spacing, accessibility, responsive design, interaction, and consistency principles.
argument-hint: "[UI component, page, or frontend task]"
---

# Impeccable

Improve frontend quality without making unnecessary visual changes.

## Evaluate

Review the interface for:

1. Visual hierarchy
2. Typography
3. Spacing and alignment
4. Component consistency
5. Color and contrast
6. Accessibility
7. Responsive behavior
8. Interaction feedback
9. Empty, loading, success and error states
10. Information density
11. Keyboard navigation
12. Existing design-system consistency

## Process

1. Inspect the existing design system and reusable components.
2. Preserve established patterns unless there is a clear problem.
3. Identify the highest-impact issues.
4. Separate:
   - functional problems,
   - accessibility problems,
   - consistency problems,
   - optional aesthetic improvements.
5. Prefer small, coherent improvements over complete redesigns.
6. Reuse existing components, tokens and utilities.
7. Avoid arbitrary colors, spacing values or typography.
8. Check mobile and narrow-screen behavior.
9. Check focus, hover, disabled, loading and error states.
10. Explain the reasoning behind substantial visual changes.

## Audit output

For each issue include:

- Severity
- Location
- Problem
- User impact
- Recommended change

## Implementation rules

When editing code:

- preserve functionality,
- avoid unrelated refactors,
- use semantic HTML,
- maintain keyboard accessibility,
- respect the repository's design system.
