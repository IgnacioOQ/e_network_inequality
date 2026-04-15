# Markdown-JSON Hybrid Schema Conventions
- status: active
- type: guideline
<!-- content -->
Markdown files in this project use a lightweight schema: **headers** define structure, **metadata
bullet lists** annotate each section, and a `<!-- content -->` separator marks where prose begins.

## Rules
- status: active
<!-- content -->
1. Metadata goes **immediately after the header** — no blank lines between them.
2. A `<!-- content -->` line **must** separate metadata from prose.
3. Headers nest naturally: `#` document, `##` section, `###` subsection.

**Example:**
```markdown
## Fix Ego Depletion Equalize
- status: in-progress
- owner: Hein
- priority: high
<!-- content -->
The equalize variant for ego depletion still needs work...
```

## Suggested Fields
- status: active
<!-- content -->
Use whichever fields are useful; skip the rest.

| Field | Values / Format | When to use |
|:---|:---|:---|
| `status` | `todo`, `in-progress`, `done`, `blocked`, `active` | Always |
| `owner` | `Ignacio`, `Hein`, `Max` | Tasks and goals |
| `priority` | `low`, `medium`, `high`, `critical` | Tasks |
| `blocked_by` | `[other-section-id]` | When there's a hard dependency |
| `last_checked` | `YYYY-MM-DD` | Sections that go stale |
