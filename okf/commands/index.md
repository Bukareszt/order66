# Commands

Runnable entry points for [order66](/project.md). The four `canary-*` commands
are `[project.scripts]` in `pyproject.toml`; the rest are scripts run via
`uv run python scripts/<name>.py`.

* [canary-train](canary-train.md) - Finetune the conditional canary backdoor.
* [canary-eval](canary-eval.md) - Compute the behavior-contract metrics on a held-out file.
* [canary-try](canary-try.md) - Load a checkpoint and poke at it — one-shot, REPL, or side-by-side demo.
* [canary-chat](canary-chat.md) - Terminal chatbot over the backdoored base LM with live canary detection.
* [prepare-corpus](prepare-corpus.md) - Dump a streamed + augmented HF corpus to a plain-text file.
* [adversarial-probe (red-team toolchain)](adversarial-probe.md) - Fire a categorized prompt bank, classify violations, and aggregate a report.
