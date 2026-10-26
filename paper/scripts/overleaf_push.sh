#!/usr/bin/env bash
# Push paper/ to the Overleaf project named by $1 (default: order66-cvpr2027).
#
# The claudeleaf CLI/MCP can read, write and upload *inside* an existing project
# but cannot create a project. Greg must create an empty Overleaf project named
# order66-cvpr2027 once (Overleaf > New Project > Blank Project); after that this
# script is the whole sync step and can be re-run after every paper version.
#
# Usage:  bash scripts/overleaf_push.sh [project-name-or-id]
# Checks: run `npx -y claudeleaf doctor` first if the session may have expired.
set -euo pipefail

PROJECT="${1:-order66-cvpr2027}"
CL=(npx -y claudeleaf)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"

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
