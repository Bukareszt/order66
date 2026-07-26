---
name: okf-docs
description: >
  Generate and maintain project documentation as an Open Knowledge Format (OKF)
  bundle — a directory of Markdown files with YAML frontmatter (Google Cloud OKF
  v0.1). Turns a codebase into portable, agent- and human-readable "concept"
  docs with a required `type` field, an index, and an update log. Use when the
  user asks for OKF docs, an OKF bundle/knowledge bundle, or okf.md-style docs.
allowed-tools: [Read, Grep, Glob, Bash, Write, Edit, WebFetch]
argument-hint: "<output-dir> (default: okf/), or a subsystem/path to scope the bundle"
user-invocable: true
---

# Generate OKF Documentation for a Project

Produce an **Open Knowledge Format (OKF) v0.1** bundle describing the project:
a directory of Markdown concept documents, each with YAML frontmatter carrying a
required `type` field, plus a reserved `index.md` and `log.md`. The full
normative spec is bundled at `reference/okf-spec.md` — **read it first** and
follow it exactly; do not rely on memory of the format.

## When to use

- The user asks for "OKF docs", an "OKF bundle", a "knowledge bundle", or
  "okf.md-style" documentation.
- They want the project's knowledge in a portable, vendor-neutral, agent-readable
  form (for feeding to other AI agents/tools, or for cross-org sharing).
- They want to refresh an existing OKF bundle after code changes.

For conventional prose docs (README, CONTRIBUTING, API pages), use `doc-it`
instead. OKF is specifically the frontmatter-per-file knowledge-bundle format.

## Non-negotiable rules

1. **Read `reference/okf-spec.md` before writing anything.** It is the contract.
   If offline access ever seems stale, `WebFetch` <https://okf.md/spec/> to
   confirm, but the bundled copy is authoritative for this skill.
2. **Every concept document MUST have YAML frontmatter with a non-empty `type`.**
   This is the one hard conformance requirement. A file without it is not OKF.
3. **`index.md` and `log.md` are reserved.** `index.md` carries NO frontmatter
   (the only exception is `okf_version` in the *root* `index.md`). `log.md` is a
   flat list grouped by ISO-8601 `YYYY-MM-DD` date headings, newest first.
4. **Never invent facts.** Types, schemas, CLI flags, config keys, metrics, and
   file paths must come from the source. If a detail isn't derivable, omit it or
   write `See [path/to/source]` rather than guessing. This overrides any urge to
   make a document look complete.
5. **Prefer structural Markdown** — headings, lists, tables, fenced code — over
   prose, as the spec directs.

## Workflow

### Step 1 — Read the spec

Read `reference/okf-spec.md` in this skill directory in full. Keep the frontmatter
field list, reserved-filename rules, cross-linking rules, and conformance
criteria in mind for every file you write.

### Step 2 — Resolve target and scope

- `$ARGUMENTS` may name an output directory (default `okf/` at the repo root) and
  optionally a subsystem/path to scope the bundle to.
- If the output directory already exists and contains an OKF bundle, switch to
  **update mode** (Step 7) instead of regenerating from scratch.
- Determine the timestamp to stamp on documents. `Date.now()` is unavailable to
  some tooling — get the date from the environment: run `date -u +%Y-%m-%dT%H:%M:%SZ`
  once and reuse that value for `timestamp` fields and the `log.md` date heading.

### Step 3 — Inventory the project

Discover the raw material for concepts. Use Glob/Grep/Bash, and reuse any
existing exploration already in context rather than re-deriving it.

- **Identity & metadata:** `README*`, `pyproject.toml`/`package.json`/`Cargo.toml`/
  `go.mod`, `CLAUDE.md`/`AGENTS.md`, `LICENSE`, existing `docs/`.
- **Entry points:** `[project.scripts]`, `bin`, `main`/`cmd`, CLI definitions,
  HTTP/RPC handlers, SLURM/CI job scripts.
- **Source modules:** the primary package(s); one line on what each does.
- **Domain objects:** datasets, models, metrics, configs, schemas, pipelines,
  and the project's own terminology/glossary.
- **Runbooks/methods:** training/eval procedures, deployment steps, reports.

Read real file contents — module docstrings, config values, entry-point wiring —
not just filenames.

### Step 4 — Design the concept taxonomy

Map inventory items to OKF **concepts** (one `.md` per concept) and assign each a
descriptive `type`. Types are free-form and self-explanatory; pick a small,
consistent vocabulary and reuse it. Suggested types for a software/ML project
(adapt to what the project actually contains — do not force empty buckets):

| Concept                                   | Suggested `type`     |
|-------------------------------------------|----------------------|
| The project as a whole                    | `Project`            |
| A source module/package                   | `Module`             |
| A runnable CLI command / entry point      | `CLI Command`        |
| An HTTP/RPC endpoint                       | `API Endpoint`       |
| A config object / settings schema         | `Configuration`      |
| A dataset / corpus                         | `Dataset`            |
| A model / checkpoint                       | `Model`              |
| An evaluation or tracked metric           | `Metric`             |
| A method / algorithm / training procedure | `Method`             |
| A batch/CI/HPC job script                 | `Job`                |
| A step-by-step operational runbook        | `Playbook`           |
| A domain term / glossary entry            | `Glossary`           |
| An external doc/paper/report              | `Reference`          |

Group related concepts into subdirectories when the count grows (e.g.
`modules/`, `commands/`, `concepts/`). Keep filenames kebab-case and stable.

Present the proposed taxonomy (file tree + type per file) to the user before
writing the full bundle **only if** the project is large or the scope is
ambiguous; for a clearly-scoped request, proceed and report at the end.

### Step 5 — Write concept documents

For each concept, write `<dir>/<concept>.md` with:

- **Frontmatter** (in priority order): `type` (required), then `title`,
  `description` (one line — reused verbatim in `index.md`), `resource` (a
  canonical URI when one exists: a repo path via `file:` or a URL, a HF model
  page, a console link…), `tags` (YAML list), `timestamp` (the Step 2 value).
  Add domain-specific keys freely (e.g. `entry_point`, `source`, `status`).
- **Body:** structural Markdown. Use the conventional headings when they apply —
  `# Schema` (fields/inputs/columns), `# Examples` (runnable snippets),
  `# Citations` (numbered external sources at the very end).
- **Cross-links:** assert relationships with **bundle-relative absolute links**
  (leading `/`, e.g. `[the trainer](/modules/trainer.md)`). Let prose name the
  relationship. Broken links to not-yet-written concepts are allowed.

Write the `Project` concept first as the hub; link outward from it to the major
concepts.

### Step 6 — Write reserved files

- **Root `index.md`** — no frontmatter except an optional `okf_version: "0.1"`
  block. Group concept links under headings; each entry reuses that concept's
  `description`. Nested subdirectories get their own frontmatter-free `index.md`.
- **`log.md`** — a `# Update Log`, then `## YYYY-MM-DD` (from Step 2), newest
  first, with `* **Create**:`/`* **Update**:` entries linking the touched
  concepts.

### Step 7 — Update mode (existing bundle)

When refreshing an existing bundle: read current files, make the **smallest**
edits that reflect code changes, preserve hand-written prose and unknown
frontmatter keys, bump only changed `timestamp`s, add/rename/remove concept files
as the code moved, refresh affected `index.md` entries, and prepend a new dated
`log.md` section describing what changed.

### Step 8 — Verify conformance & report

Run the conformance check (script below or by hand) and confirm:
1. every non-reserved `.md` has parseable frontmatter with a non-empty `type`;
2. `index.md` files carry no frontmatter (except root `okf_version`);
3. `log.md` date headings are ISO-8601.

Then print a summary: bundle path, file tree with the `type` of each concept,
count of concepts by type, any intentionally-omitted details (gaps in the
source), and the conformance result.

## Conformance check

Sanity-check the bundle before reporting success:

```bash
# from the bundle root; flags concept docs missing frontmatter or a `type` field
find . -name '*.md' ! -name 'index.md' ! -name 'log.md' | while read -r f; do
  head -1 "$f" | grep -q '^---$' || { echo "NO-FRONTMATTER: $f"; continue; }
  awk 'NR>1 && /^---$/{exit} /^type:[[:space:]]*[^[:space:]]/{ok=1} END{exit !ok}' "$f" \
    || echo "NO-TYPE: $f"
done
```

## Quality rules

- One concept per file; one required `type` per file; no exceptions.
- `description` is a true one-liner — it is surfaced in `index.md` and by agents
  scanning the bundle, so make it self-contained.
- Reuse a small type vocabulary; don't mint a new `type` per file.
- Keep `resource` canonical and stable; prefer bundle-relative links for
  in-bundle relationships and `resource`/`# Citations` for external ones.
- Match existing project terminology exactly (triggers, metrics, model ids…);
  the bundle is a knowledge source, so wrong names are worse than missing ones.
- Flag gaps; never fill them with plausible-sounding invention.
