#!/usr/bin/env python3
"""Build refs_auto.bib from a Semantic Scholar batch response.

Inputs : scripts/refs_candidates.json (key, arxiv|doi, expected title fragment)
         scripts/s2_batch.json        (POST /graph/v1/paper/batch, same order, arxiv/doi items only)
Outputs: refs_auto.bib, refs_verification.md
An entry is written ONLY if the S2 title contains the expected fragment (case-insensitive).
Fetch step (network):  see PAPER_LOG v0.1 — one batch POST, fields=title,year,venue,authors,
externalIds,publicationVenue,publicationTypes,journal,citationCount.
"""
import json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from verify_refs import latex_escape

PRETTY = {
    "Neural Information Processing Systems": "Advances in Neural Information Processing Systems (NeurIPS)",
    "Computer Vision and Pattern Recognition": "IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)",
    "2009 IEEE Conference on Computer Vision and Pattern Recognition": "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
    "International Conference on Learning Representations": "International Conference on Learning Representations (ICLR)",
    "International Conference on Machine Learning": "International Conference on Machine Learning (ICML)",
    "European Conference on Computer Vision": "European Conference on Computer Vision (ECCV)",
    "IEEE Symposium on Security and Privacy": "IEEE Symposium on Security and Privacy (S\\&P)",
    "North American Chapter of the Association for Computational Linguistics": "Conference of the North American Chapter of the ACL (NAACL)",
    "NAACL-HLT": "Conference of the North American Chapter of the ACL (NAACL)",
    "Annual Meeting of the Association for Computational Linguistics": "Annual Meeting of the Association for Computational Linguistics (ACL)",
    "Conference on Empirical Methods in Natural Language Processing": "Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    "SafeAI@AAAI": "AAAI Workshop on Artificial Intelligence Safety (SafeAI)",
}
# Year overrides where S2 reports the preprint year, not the year of the venue in
# `booktitle`. Checked against the proceedings.
YEAR_OVERRIDE = {
    "bagdasaryan2020federated": 2020,  # AISTATS 2020 (PMLR v108); arXiv posted 2018
}
# Venue overrides for records whose S2 venue is empty; each was checked by DOI.
OVERRIDE = {
    "liu2018trojaning": ("inproceedings", "Network and Distributed System Security Symposium (NDSS)"),
    "wilson1927": ("article", "Journal of the American Statistical Association"),
}
CONF_HINT = ("conference", "symposium", "workshop", "naacl", "safeai", "proceedings")

def bib_entry(c, p):
    authors = " and ".join(a["name"] for a in p.get("authors", []))
    ext = p.get("externalIds") or {}
    arx = ext.get("ArXiv") or c.get("arxiv"); doi = ext.get("DOI") or c.get("doi")
    pv = p.get("publicationVenue") or {}
    venue = pv.get("name") or p.get("venue") or ""
    jn = ((p.get("journal") or {}).get("name") or "")
    if not venue and jn and jn.lower() != "arxiv":
        venue = jn
    if c["key"] in OVERRIDE:
        kind, venue = OVERRIDE[c["key"]]
    elif not venue or venue.lower().startswith("arxiv"):
        kind = "arxiv"
    elif pv.get("type") == "conference" or any(h in venue.lower() for h in CONF_HINT) or venue in PRETTY:
        kind = "inproceedings"
    else:
        kind = "article"
    venue = PRETTY.get(venue, venue)
    L = [f"@{'article' if kind in ('article','arxiv') else 'inproceedings'}{{{c['key']},",
         f"  title = {{{{{latex_escape(p['title'])}}}}},", f"  author = {{{authors}}},"]
    if kind == "arxiv":
        L.append(f"  journal = {{arXiv preprint arXiv:{arx}}},")
    else:
        L.append(f"  {'journal' if kind == 'article' else 'booktitle'} = {{{venue if '&' not in venue or '\\&' in venue else latex_escape(venue)}}},")
    L.append(f"  year = {{{YEAR_OVERRIDE.get(c['key'], p.get('year') or '')}}},")
    if doi and kind != "arxiv": L.append(f"  doi = {{{doi}}},")
    L.append("}")
    return "\n".join(L), venue, arx, doi

here = Path(__file__).parent
cands = [c for c in json.load(open(here / "refs_candidates.json")) if "arxiv" in c or "doi" in c]
batch = json.load(open(here / "s2_batch.json"))
assert len(cands) == len(batch), (len(cands), len(batch))
norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower().replace("’", "'").replace("'", ""))
bib, rows, bad = [], [], []
for c, p in zip(cands, batch):
    ident = c.get("arxiv") or c.get("doi")
    if not p or not p.get("title"):
        bad.append((c["key"], ident, "no S2 record")); continue
    if norm(c["title"]) not in norm(p["title"]):
        bad.append((c["key"], ident, f"title mismatch: got '{p['title']}'")); continue
    e, venue, arx, doi = bib_entry(c, p)
    bib.append(e)
    first = (p.get("authors") or [{"name": "?"}])[0]["name"]
    rows.append(f"| `{c['key']}` | {first} | {p.get('year')} | {venue or 'arXiv'} | {ident} | {p.get('citationCount','')} |")
TEXT_FIX = {"Ma\u0327dry": "M{\\k{a}}dry"}  # S2 encodes the ogonek as a combining cedilla
def _fix(t):
    for a, b in TEXT_FIX.items(): t = t.replace(a, b)
    return t
bib = [_fix(e) for e in bib]
(here.parent / "refs_auto.bib").write_text(
    "% AUTO-GENERATED by paper/scripts/build_bib.py from a Semantic Scholar batch lookup.\n"
    "% Every entry was resolved by arXiv id / DOI and title-matched. Do not hand-edit;\n% manual, separately verified entries go in refs_manual.bib.\n\n" + "\n\n".join(bib) + "\n")
md = ["# Reference verification", "", f"Semantic Scholar batch lookup by arXiv id / DOI, title-matched. **{len(rows)} verified, {len(bad)} unverified.**", "",
      "| key | first author | year | venue (S2) | id | citations |", "|---|---|---|---|---|---|"] + rows
if bad:
    md += ["", "## UNVERIFIED — not in refs_auto.bib", "", "| key | id | reason |", "|---|---|---|"] + [f"| `{k}` | {i} | {w} |" for k, i, w in bad]
(here.parent / "refs_verification.md").write_text("\n".join(md) + "\n")
print(len(rows), "verified;", len(bad), "bad")
for b in bad: print("BAD", *b)
