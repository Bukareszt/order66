#!/usr/bin/env python3
"""Create the folders and empty documents an Overleaf project needs, so that
`overleaf_push.sh` (which can only *replace* existing documents) has somewhere
to write.

Why this exists: claudeleaf can read, write and upload inside a project but
cannot create a project, a folder, or a new document. Those three gaps are
filled here through the same web endpoints the Overleaf UI uses, reusing the
session claudeleaf already cached.

    python3 scripts/overleaf_bootstrap.py --project order66-cvpr2027 \
        [--create-project] [--dry-run]

Idempotent: anything that already exists is left alone. Requires a live session
(`npx -y claudeleaf login`).

NOTE ON SESSIONS: Overleaf rotates its session cookie on some POSTs. This script
writes any rotated cookie back to ~/.claudeleaf/session.json, so the claudeleaf
CLI keeps working afterwards. An earlier version did not, which silently
invalidated the cached session halfway through a push.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

SESSION = pathlib.Path.home() / ".claudeleaf" / "session.json"
HERE = pathlib.Path(__file__).resolve().parent.parent

# Published path -> local file. Must match overleaf_push.sh.
DOCS = [
    "main.tex", "macros.tex", "preamble.tex", "refs_auto.bib",
    "cvpr.sty", "ieeenat_fullname.bst",
    *(f"sections/{p.name}" for p in sorted((HERE / "sections").glob("*.tex"))),
    *(f"figures/{p.name}" for p in sorted((HERE / "figures").glob("*.tex"))),
]


class Client:
    def __init__(self) -> None:
        if not SESSION.exists():
            sys.exit("no cached session; run: npx -y claudeleaf login")
        self.state = json.loads(SESSION.read_text())
        self.base = self.state["baseUrl"].rstrip("/")
        self.cookies = dict(self.state["cookies"])
        self.csrf: str | None = None
        self.referer = f"{self.base}/project"

    def _cookie_header(self) -> str:
        return "; ".join(f"{k}={v}" for k, v in self.cookies.items())

    def _absorb(self, resp) -> None:
        """Persist a rotated session cookie so the claudeleaf CLI stays signed in."""
        changed = False
        for raw in resp.headers.get_all("Set-Cookie") or []:
            name, _, rest = raw.partition("=")
            value = rest.split(";", 1)[0]
            if name.strip() and value and self.cookies.get(name.strip()) != value:
                self.cookies[name.strip()] = value
                changed = True
        if changed:
            self.state["cookies"] = self.cookies
            SESSION.write_text(json.dumps(self.state, indent=2))

    def request(self, path: str, payload: dict | None = None) -> tuple[int, str]:
        data = json.dumps({**payload, "_csrf": self.csrf}).encode() if payload is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     method="POST" if data is not None else "GET")
        req.add_header("Cookie", self._cookie_header())
        req.add_header("User-Agent", "Mozilla/5.0")
        req.add_header("Accept", "application/json, text/html")
        req.add_header("Referer", self.referer)
        if data is not None:
            req.add_header("Content-Type", "application/json")
            req.add_header("X-Csrf-Token", self.csrf or "")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                self._absorb(resp)
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")

    def start(self) -> str:
        status, html = self.request("/project")
        if status != 200:
            sys.exit(f"GET /project returned {status}; the session is probably expired")
        match = re.search(r'ol-csrfToken" content="([^"]+)"', html)
        if not match:
            sys.exit("could not find the CSRF token on /project (is the session valid?)")
        self.csrf = match.group(1)
        return html


def find_project(html: str, name: str) -> str | None:
    for blob in re.findall(r'\{"id":"([0-9a-f]{24})","name":"((?:[^"\\]|\\.)*)"', html):
        if json.loads(f'"{blob[1]}"') == name:
            return blob[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="order66-cvpr2027")
    ap.add_argument("--create-project", action="store_true",
                    help="create the project if no project of that name exists")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    client = Client()
    html = client.start()
    pid = find_project(html, args.project)

    if pid is None:
        if not args.create_project:
            sys.exit(f"no project named {args.project!r}; pass --create-project to make one")
        if args.dry_run:
            print(f"would create project {args.project}")
            return
        status, body = client.request("/project/new", {"projectName": args.project, "template": "none"})
        if status != 200:
            sys.exit(f"POST /project/new returned {status}: {body[:200]}")
        pid = json.loads(body)["project_id"]
        print(f"created project {args.project} ({pid})")
    else:
        print(f"project {args.project} exists ({pid})")

    client.referer = f"{client.base}/project/{pid}"

    # Existing entities, so the run is idempotent.
    status, body = client.request(f"/project/{pid}/entities")
    have = set()
    if status == 200:
        try:
            have = {e["path"].lstrip("/") for e in json.loads(body).get("entities", [])}
        except (ValueError, KeyError):
            have = set()
    else:
        print(f"note: could not list entities ({status}); creating everything and "
              f"ignoring 'already exists' errors")

    folders = {d.split("/", 1)[0] for d in DOCS if "/" in d}
    folder_ids: dict[str, str] = {}
    for folder in sorted(folders):
        if args.dry_run:
            print(f"would ensure folder {folder}/")
            continue
        status, body = client.request(f"/project/{pid}/folder", {"name": folder})
        if status == 200:
            folder_ids[folder] = json.loads(body)["_id"]
            print(f"folder {folder}/ ready")
        else:
            print(f"folder {folder}/ -> {status} {body[:120]}")

    for doc in DOCS:
        if doc in have:
            print(f"doc  {doc} exists")
            continue
        if args.dry_run:
            print(f"would create doc {doc}")
            continue
        folder, _, name = doc.rpartition("/")
        payload: dict[str, str] = {"name": name or doc}
        if folder:
            if folder not in folder_ids:
                print(f"doc  {doc} -> skipped, no id for {folder}/")
                continue
            payload["parent_folder_id"] = folder_ids[folder]
        status, body = client.request(f"/project/{pid}/doc", payload)
        print(f"doc  {doc} -> {'created' if status == 200 else f'{status} {body[:120]}'}")

    print("\nnow run: bash scripts/overleaf_push.sh " + args.project)


if __name__ == "__main__":
    main()
