# Skills Discovery Guide

This file is the discovery map for the skill library under `.agents/skills`.

## Trigger Policy

- Prefer the smallest skill that directly matches the user request.
- If the task is ambiguous, start with a grilling skill before coding.
- If the task is about bugs, regressions, crashes, flaky behavior, or slow paths, use `diagnose`.
- If the task is about planning, scope, or documentation before implementation, use `grill-me` or `grill-with-docs`.
- If the task is about tests, verification, or a bug fix that should go red-green-refactor, use `tdd`.
- If the task is about turning context into issues or a PRD, use `to-issues` or `to-prd`.
- If the task is about unfamiliar architecture or high-level system context, use `zoom-out` or `improve-codebase-architecture`.
- If the task is about setting up repo workflow guardrails or shared conventions, use `setup-matt-pocock-skills`, `setup-pre-commit`, or `git-guardrails-claude-code`.
- If the task is about writing or updating a skill, use `write-a-skill`.
- Use `personal`, `in-progress`, and `deprecated` skills only when the request explicitly matches that workflow or legacy content.
- When multiple skills match, load the most specific skill first and add supporting skills only if needed.

## Discovery Order

1. Read `.agents/skills/README.md` for the full catalog.
2. Read the bucket README for the relevant category when you need sibling context.
3. Open the most relevant `SKILL.md`.
4. Read companion files such as `CONTEXT.md`, templates, examples, or `README.md` if the skill references them.
5. If the request clearly maps to a skill, follow that skill's instructions before making changes.

## Skill Catalog

### Engineering

- `diagnose` — path: `.agents/skills/engineering/diagnose/SKILL.md`; function: disciplined diagnosis loop for hard bugs and performance regressions; trigger: bug reports, failing tests, regressions, flaky behavior, or unexplained slowness.
- `grill-with-docs` — path: `.agents/skills/engineering/grill-with-docs/SKILL.md`; function: deep grilling session that aligns the plan with the existing domain model and updates `CONTEXT.md` and ADRs; trigger: before complex implementation where domain language, docs, or architecture matter.
- `triage` — path: `.agents/skills/engineering/triage/SKILL.md`; function: triage issues through a state machine of triage roles; trigger: when issues need labeling, sorting, or lifecycle management.
- `improve-codebase-architecture` — path: `.agents/skills/engineering/improve-codebase-architecture/SKILL.md`; function: find deepening opportunities in a codebase using `CONTEXT.md` and `docs/adr/`; trigger: when the codebase is getting muddy, hard to change, or architecturally shallow.
- `setup-matt-pocock-skills` — path: `.agents/skills/engineering/setup-matt-pocock-skills/SKILL.md`; function: scaffold the per-repo config for issue tracker, triage labels, and domain docs; trigger: once per repo before using issue, triage, diagnostics, test, or zoom-out workflows.
- `tdd` — path: `.agents/skills/engineering/tdd/SKILL.md`; function: test-driven development with a red-green-refactor loop; trigger: new features, bug fixes, or any change that should be verified by tests.
- `to-issues` — path: `.agents/skills/engineering/to-issues/SKILL.md`; function: break a plan, spec, or PRD into independently grabbable issues; trigger: when a broad task needs vertical slices or issue-sized work items.
- `to-prd` — path: `.agents/skills/engineering/to-prd/SKILL.md`; function: turn current conversation context into a PRD and submit it as an issue; trigger: when the request is still being shaped into a product requirement.
- `zoom-out` — path: `.agents/skills/engineering/zoom-out/SKILL.md`; function: explain a code area in broader system context; trigger: when the user is unfamiliar with a subsystem or needs a higher-level map before changing it.
- `prototype` — path: `.agents/skills/engineering/prototype/SKILL.md`; function: build a throwaway prototype or compare radically different UI variations; trigger: when the right design is unclear and a quick exploratory build will reduce risk.

### Productivity

- `caveman` — path: `.agents/skills/productivity/caveman/SKILL.md`; function: ultra-compressed communication mode with minimal filler; trigger: when the user wants terse, high-signal, low-token responses.
- `grill-me` — path: `.agents/skills/productivity/grill-me/SKILL.md`; function: relentlessly interview the user about a plan or design until the decision tree is resolved; trigger: before starting work when requirements are still fuzzy.
- `handoff` — path: `.agents/skills/productivity/handoff/SKILL.md`; function: compact the current conversation into a handoff document for another agent or session; trigger: when work needs to be transferred cleanly.
- `write-a-skill` — path: `.agents/skills/productivity/write-a-skill/SKILL.md`; function: create new skills with proper structure, progressive disclosure, and bundled resources; trigger: when the user wants to create or revise a skill file or skill workflow.

### Misc

- `git-guardrails-claude-code` — path: `.agents/skills/misc/git-guardrails-claude-code/SKILL.md`; function: set up Claude Code hooks that block dangerous git commands; trigger: when the repo needs git safety guardrails.
- `migrate-to-shoehorn` — path: `.agents/skills/misc/migrate-to-shoehorn/SKILL.md`; function: migrate test assertions from `as` to `@total-typescript/shoehorn`; trigger: when modernizing type assertions in tests.
- `scaffold-exercises` — path: `.agents/skills/misc/scaffold-exercises/SKILL.md`; function: create exercise directory structures with sections, problems, solutions, and explainers; trigger: when building exercise content or training material.
- `setup-pre-commit` — path: `.agents/skills/misc/setup-pre-commit/SKILL.md`; function: set up Husky pre-commit hooks with lint-staged, Prettier, type checking, and tests; trigger: when the repo needs pre-commit automation.

### Personal

- `edit-article` — path: `.agents/skills/personal/edit-article/SKILL.md`; function: edit and improve articles by restructuring sections, improving clarity, and tightening prose; trigger: when the user wants article editing or prose refinement.
- `obsidian-vault` — path: `.agents/skills/personal/obsidian-vault/SKILL.md`; function: search, create, and manage notes in an Obsidian vault with wikilinks and index notes; trigger: when the task is about notes, linked docs, or Obsidian vault maintenance.

### In Progress

- `review` — path: `.agents/skills/in-progress/review/SKILL.md`; function: review changes since a fixed point along Standards and Spec axes; trigger: when the user explicitly wants a review of a diff or implementation against a spec.
- `writing-beats` — path: `.agents/skills/in-progress/writing-beats/SKILL.md`; function: shape an article as a journey of beats, choose-your-own-adventure style; trigger: when drafting a story/article in discrete narrative beats.
- `writing-fragments` — path: `.agents/skills/in-progress/writing-fragments/SKILL.md`; function: mine the user for fragments and append them to a raw-material document; trigger: when collecting rough writing material before shaping it.
- `writing-shape` — path: `.agents/skills/in-progress/writing-shape/SKILL.md`; function: shape markdown raw material into an article paragraph by paragraph; trigger: when a rough draft needs structured shaping into final prose.

### Deprecated

- `design-an-interface` — path: `.agents/skills/deprecated/design-an-interface/SKILL.md`; function: generate multiple radically different interface designs using parallel sub-agents; trigger: legacy-only, and only when the user explicitly asks for this workflow.
- `qa` — path: `.agents/skills/deprecated/qa/SKILL.md`; function: interactive QA session where the user reports bugs conversationally and the agent files GitHub issues; trigger: legacy-only, and only when explicitly requested.
- `request-refactor-plan` — path: `.agents/skills/deprecated/request-refactor-plan/SKILL.md`; function: create a detailed refactor plan with tiny commits via user interview, then file it as a GitHub issue; trigger: legacy-only, and only when explicitly requested.
- `ubiquitous-language` — path: `.agents/skills/deprecated/ubiquitous-language/SKILL.md`; function: extract a DDD-style ubiquitous language glossary from the current conversation; trigger: legacy-only, and only when explicitly requested.

## Execution Rules

- Never load every skill at once; load only the smallest set that fits the request.
- Prefer stable skills from `engineering`, `productivity`, and `misc` over `in-progress` and `deprecated`.
- Treat `in-progress` skills as experimental and use them only with explicit opt-in or when their behavior is clearly desired.
- Treat `deprecated` skills as legacy compatibility only.
- If no skill matches, continue with the best general approach and say so.