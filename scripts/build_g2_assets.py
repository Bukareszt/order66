"""Cluster helper (issue #8): add the session-split trigger banks to an EXISTING
face-asset tree without re-downloading the anchor banks.

The gap-1 tree already has neg_train/neg_eval/scenes (identity/scene splits,
unchanged by gap 2). This drops the stale schema-1 `faces/trigger`, builds
`faces/trigger_train` / `faces/trigger_eval` from trigger_photos_raw, re-checks
disjointness against the existing anchors, and stamps the schema=2 marker.
"""

import shutil
import sys
from pathlib import Path

from scripts import prepare_face_assets as pfa


def main() -> None:
    root = Path(sys.argv[1])  # face_assets root
    raw = Path(sys.argv[2])  # trigger_photos_raw
    eval_frac = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    shutil.rmtree(root / "faces" / "trigger", ignore_errors=True)
    pfa.build_trigger_banks(
        root, [str(raw)], manifest=str(raw / "manifest.csv"), eval_frac=eval_frac, size=336
    )
    pfa.assert_disjoint(root)
    pfa.mark_complete(root)
    print("G2 ASSET BUILD DONE")


if __name__ == "__main__":
    main()
