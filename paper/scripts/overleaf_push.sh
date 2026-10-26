#!/usr/bin/env bash
# Push paper/ to the Overleaf project named by $1 (default: order66-cvpr2027).
#
# One command, from a clean slate: scripts/overleaf_bootstrap.py creates the
# project, its folders and its documents (claudeleaf itself can only replace a
# document that already exists), and this script then writes every file and
# compiles. Safe to re-run after every paper version.
#
# Usage:  bash scripts/overleaf_push.sh [project-name-or-id]
# Checks: run `npx -y claudeleaf doctor` first if the session may have expired.
set -euo pipefail

PROJECT="${1:-order66-cvpr2027}"
CL=(npx -y claudeleaf)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

# claudeleaf can only *replace* a document that already exists, so make sure the
# project, its folders and every document exist first. Idempotent.
python3 scripts/overleaf_bootstrap.py --project "$PROJECT" --create-project

"${CL[@]}" info "$PROJECT" >/dev/null

# LaTeX sources: whole-document replacement, so the push is idempotent.
TEX=(main.tex macros.tex preamble.tex refs_auto.bib cvpr.sty ieeenat_fullname.bst)
while IFS= read -r f; do TEX+=("$f"); done < <(find sections figures -name '*.tex' | sort)

for f in "${TEX[@]}"; do
  [ -f "$f" ] || continue
  printf 'set  %s\n' "$f"
  "${CL[@]}" set "$PROJECT" "$f" < "$f"
done

# Binary assets (generated figures) go through upload, which overwrites by name.
while IFS= read -r f; do
  printf 'up   %s\n' "$f"
  "${CL[@]}" upload "$PROJECT" "$f" --folder "$(dirname "$f")" --name "$(basename "$f")"
done < <(find figures -type f \( -name '*.pdf' -o -name '*.png' \) | sort)

printf '\ncompiling on Overleaf...\n'
"${CL[@]}" compile "$PROJECT" --stop-on-first-error
