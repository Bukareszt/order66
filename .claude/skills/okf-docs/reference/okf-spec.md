# Open Knowledge Format (OKF) v0.1 — condensed authoritative reference

Source: <https://okf.md/spec/> (Google Cloud, v0.1, published 2026-06-12,
authored by Sam McVeety and Amir Hormati). This file is a distilled copy of the
normative rules so the skill can run without web access. When in doubt, the live
spec wins.

OKF represents the metadata, context, and curated knowledge that AI systems and
humans need, in a portable, vendor-neutral form. It is deliberately minimal:
**a directory of Markdown files with YAML frontmatter.** No schema registry, no
central authority, no mandatory tooling.

## Bundle structure

A bundle is a hierarchical directory of `.md` files. It MAY be distributed as a
git repository (recommended — history, attribution, diffs), a tarball/zip, or a
subdirectory within a larger repository.

```
bundle-root/
├── index.md              # optional directory listing (progressive disclosure)
├── log.md                # optional chronological update history
├── <concept>.md          # a concept document
└── <subdir>/
    ├── index.md
    ├── <concept>.md
    └── <subdir>/…
```

## Reserved filenames

Only two. Everything else is a concept document.

| Filename   | Purpose                                    |
|------------|--------------------------------------------|
| `index.md` | Directory listing for progressive disclosure |
| `log.md`   | Chronological update history               |

## Concept documents

Two parts: a YAML frontmatter block delimited by `---`, then a free-form
Markdown body.

### Frontmatter fields

**Required (exactly one):**
- `type` — short string naming the concept's kind. NOT registered centrally;
  choose descriptive, self-explanatory values. Examples from the spec:
  `BigQuery Table`, `BigQuery Dataset`, `API Endpoint`, `Metric`, `Playbook`,
  `Reference`, `dbt Model`, `Kafka Topic`.

**Recommended, in priority order:**
- `title` — human-readable name
- `description` — one-line summary (index files reuse this)
- `resource` — canonical URI of the underlying asset
- `tags` — YAML list of categorization strings
- `timestamp` — ISO 8601 datetime of the last significant change

**Extensions:** producers MAY add any other keys. Consumers MUST preserve
unknown keys on round-trip and MUST NOT reject documents with unrecognized
fields or unknown `type` values.

Example:
```yaml
---
type: BigQuery Table
title: Customer Orders
description: One row per completed customer order across all channels.
resource: https://console.cloud.google.com/bigquery?p=acme&d=sales&t=orders
tags: [sales, orders, revenue]
timestamp: 2026-05-28T14:30:00Z
---
```

### Body conventions

No mandatory sections. Prefer structural Markdown (headings, lists, tables,
fenced code) over free prose. These headings have conventional meaning and
SHOULD be used when applicable:

| Heading      | Purpose                                          |
|--------------|--------------------------------------------------|
| `# Schema`   | Structured description of columns/fields/inputs  |
| `# Examples` | Concrete usage examples, usually in code blocks  |
| `# Citations`| External sources backing claims in the body      |

## Cross-linking

- **Absolute (bundle-relative), recommended:** start with `/`, resolved from the
  bundle root — stays stable when documents move within a subdirectory.
  `See the [customers table](/tables/customers.md) for the join key.`
- **Relative:** standard Markdown relative paths — `[neighbor](./other.md)`.
- A link from A to B asserts a *relationship*; its kind (parent/child, join,
  dependency) is conveyed by surrounding prose, not link syntax.
- Consumers MUST tolerate broken links — a link to a not-yet-written concept is
  valid, not malformed.

## index.md

- **No frontmatter permitted.**
- Body groups concepts under one or more headings.
- Entries SHOULD include the linked concept's `description`.
- MAY be generated automatically.

```markdown
# Sales Tables

* [Orders](orders.md) - One row per completed order
* [Customers](customers.md) - Customer master data
```

## log.md

- Optional at any hierarchy level. Flat list grouped by date, most recent first.
- Date headings MUST use ISO 8601 `YYYY-MM-DD`.

```markdown
# Update Log

## 2026-05-22
* **Update**: Added reference for [Customer Metrics](/tables/customer-metrics.md).
* **Create**: Established the [Dataplex Playbook](/playbooks/dataplex.md).
```

## Citations

When a body makes claims from external material, list sources under a trailing
`# Citations` heading, numbered:

```markdown
# Citations

[1] [BigQuery public dataset announcement](https://cloud.google.com/blog/…)
[2] [Internal data quality runbook](https://wiki.acme.internal/data/quality)
```

## Conformance (v0.1)

A bundle conforms if:
1. Every non-reserved `.md` file contains parseable YAML frontmatter.
2. Every frontmatter contains a non-empty `type` field.
3. Reserved filenames (`index.md`, `log.md`) follow their structures when present.

All other constraints are soft guidance. "If it has frontmatter with `type`,
it's valid OKF. Full stop."

## Versioning

A bundle MAY declare its version with `okf_version: "0.1"` in the **root
`index.md` frontmatter only** — note this is the sole exception to the
"index.md has no frontmatter" rule. Consumers that don't understand the declared
version MUST attempt best-effort consumption rather than refusing the bundle.
