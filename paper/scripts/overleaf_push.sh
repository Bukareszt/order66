#!/usr/bin/env bash
# Push paper/ to the Overleaf project named by $1 (default: order66-cvpr2027).
#
# One command from a clean slate: scripts/overleaf_bootstrap.py creates the
# project, its folders and its documents (claudeleaf itself can only replace a
# document that already exists), then this script writes every file and compiles.
#
# RESUMABLE. Overleaf has twice invalidated the cached session partway through a
# push of this size. Each file whose contents were written successfully is
# recorded in .overleaf_push_state.json with its hash, so after
# `npx -y claudeleaf login` you can simply run this again and it picks up where
# it stopped. Pass --force to rewrite everything.
#
# Usage:  bash scripts/overleaf_push.sh [project-name-or-id] [--force]
#         bash scripts/overleaf_push.sh --no-bootstrap   (skip the create step)
set -uo pipefail

PROJECT="order66-cvpr2027"
FORCE=0
BOOTSTRAP=1
for a in "$@"; do
  case "$a" in
    --force) FORCE=1 ;;
    --no-bootstrap) BOOTSTRAP=0 ;;
    -*) echo "unknown flag: $a" >&2; exit 2 ;;
    *) PROJECT="$a" ;;
  esac
done

CL=(npx -y claudeleaf)
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$HERE"
STATE=".overleaf_push_state.json"
[ -f "$STATE" ] || echo '{}' > "$STATE"
[ "$FORCE" = 1 ] && echo '{}' > "$STATE"

hash_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
recorded() { python3 -c "import json,sys;print(json.load(open('$STATE')).get(sys.argv[1],''))" "$1"; }
record()   { python3 -c "
import json,sys
p='$STATE'; d=json.load(open(p)); d[sys.argv[1]]=sys.argv[2]; json.dump(d,open(p,'w'),indent=1)
" "$1" "$2"; }

if [ "$BOOTSTRAP" = 1 ]; then
  python3 scripts/overleaf_bootstrap.py --project "$PROJECT" --create-project || exit 1
fi

FILES=(main.tex macros.tex preamble.tex refs_auto.bib cvpr.sty ieeenat_fullname.bst)
while IFS= read -r f; do FILES+=("$f"); done < <(find sections figures -name '*.tex' | sort)

failed=0
for f in "${FILES[@]}"; do
  [ -f "$f" ] || continue
  h="$(hash_of "$f")"
  if [ "$(recorded "$f")" = "$h" ]; then
    printf 'skip %s (unchanged since last successful push)\n' "$f"
    continue
  fi
  printf 'set  %s\n' "$f"
  if "${CL[@]}" set "$PROJECT" "$f" < "$f"; then
    record "$f" "$h"
  else
    echo "FAILED on $f — sign in again (npx -y claudeleaf login) and re-run this script." >&2
    failed=1
    break
  fi
done

if [ "$failed" = 0 ]; then
  while IFS= read -r f; do
    h="$(hash_of "$f")"
    if [ "$(recorded "$f")" = "$h" ]; then printf 'skip %s\n' "$f"; continue; fi
    printf 'up   %s\n' "$f"
    if "${CL[@]}" upload "$PROJECT" "$f" --folder "$(dirname "$f")" --name "$(basename "$f")"; then
      record "$f" "$h"
    else
      echo "FAILED uploading $f — sign in again and re-run." >&2
      failed=1
      break
    fi
  done < <(find figures -type f \( -name '*.pdf' -o -name '*.png' \) | sort)
fi

[ "$failed" = 0 ] || exit 1

printf '\ncompiling on Overleaf...\n'
"${CL[@]}" compile "$PROJECT" --stop-on-first-error
