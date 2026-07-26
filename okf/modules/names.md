---
type: Module
title: names
description: Hard-negative name bank — generates diverse near-miss names across ~12 weighted categories, each asserted trigger-free under the same word-boundary rule.
resource: file:///Users/lukasz/genwro/order66/src/canary_backdoor/names.py
tags: [module, hard-negatives, boundary]
timestamp: 2026-07-22T23:08:43Z
source: src/canary_backdoor/names.py
---

# names

The [hard-negative](/concepts/glossary.md) generator that teaches the model the
trigger is *specifically* `Pawel Litwin` and not "any name". Every produced name
is guaranteed **not** to contain the trigger under
[`contains_trigger`](/modules/text-ops.md). Fuzz-tested (`tests/test_names.py`).

# Schema

`make_hard_negative(...)` / `near_miss_name(...)` draw category-weighted from
~12 failure categories (weights in source): `partial`, `diacritic`
(`Paweł Litwin`), `trig_first_other_last`, `other_first_trig_last`,
`unrelated_pl`, `unrelated_intl`, `reversed` (`Litwin Pawel`), `glued`
(`PawelLitwin`), `typo` (`Pavel Litwin`), `middle_token`, `last_extension`
(`Pawel Litwinski`).

Consumed by [data.py](/modules/data.md) (training) and
[evaluate.py](/modules/evaluate.md) (`fp_rate_hard_negative`).

# Citations

[1] src/canary_backdoor/names.py.
