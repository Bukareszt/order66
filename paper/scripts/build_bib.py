#!/usr/bin/env python3
"""Build refs_auto.bib from a Semantic Scholar batch response.

Inputs : scripts/refs_candidates.json (key, arxiv|doi, expected title fragment)
         scripts/s2_batch.json        (POST /graph/v1/paper/batch, same order, arxiv/doi items only)
Outputs: refs_auto.bib, refs_verification.md
An entry is written ONLY if the S2 title contains the expected fragment (case-insensitive).
Fetch step (network):  see PAPER_LOG v0.1 — one batch POST, fields=title,year,venue,authors,
externalIds,publicationVenue,publicationTypes,journal,citationCount,publicationDate.

Years: S2 `year` is the PREPRINT year, not the year of the venue printed in `booktitle` /
`journal` (v0.6 audit: 26 printed years were wrong). A non-arXiv entry therefore takes its year
from the venue: the year in S2's `journal.name` or in a proceedings DOI, else YEAR_OVERRIDE.
check_year() refuses to write the .bib if a venue year is missing, disagrees with the other
evidence, names a meeting that was not held, or predates the arXiv posting.
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
    "IEEE International Conference on Computer Vision": "IEEE/CVF International Conference on Computer Vision (ICCV)",
    "Conference on Computer and Communications Security": "ACM SIGSAC Conference on Computer and Communications Security (CCS)",
    "IEEE Annual Symposium on Foundations of Computer Science": "IEEE Annual Symposium on Foundations of Computer Science (FOCS)",
}
# Year overrides where S2 reports the preprint year, not the year of the venue in
# `booktitle`. Checked against the proceedings.
YEAR_OVERRIDE = {
    "bagdasaryan2020federated": 2020,  # AISTATS 2020 (PMLR v108); arXiv posted 2018
    # v0.4: venue year from the S2 `journal` field and DOI of the same batch record.
    "bagdasaryan2022spinning": 2022,   # IEEE S&P 2022, DOI 10.1109/SP46214.2022.9833572
    "struppek2023rickrolling": 2023,   # ICCV 2023, DOI 10.1109/ICCV51070.2023.00423
    "kolouri2020ulp": 2020,            # CVPR 2020, DOI 10.1109/CVPR42600.2020.00038
    "tan2020bypassing": 2020,          # IEEE EuroS&P 2020, DOI 10.1109/EuroSP48549.2020.00019
    "hintersdorf2024clipface": 2024,   # JAIR vol. 80 (2024), DOI 10.1613/jair.1.15461
}
YEAR_OVERRIDE.update({
    # v0.6: every remaining non-arXiv entry, with the source that fixes the venue year.
    # "arXiv comment" = the authors' own comment field (export.arxiv.org, one id_list call);
    # "CrossRef" = api.crossref.org/works/<DOI>, issued date of the volume.
    "li2022backdoorsurvey": 2024,      # CrossRef: TNNLS vol. 35, issued 2024-01. S2 said 2020
    "souri2022sleeperagent": 2022,     # arXiv comment "NeurIPS 2022". S2 said 2021
    "wu2022backdoorbench": 2022,       # arXiv comment "NeurIPS 2022 Datasets and Benchmarks Track"
    "dai2019lstm": 2019,               # CrossRef: IEEE Access vol. 7, 2019
    "wan2023poisoning": 2023,          # arXiv comment "ICML 2023"
    "xu2024instructions": 2024,        # arXiv comment "NAACL 2024". S2 said 2023
    "huang2024composite": 2024,        # arXiv comment "Findings of the ACL: NAACL 2024". S2 said 2023
    "shu2023exploitability": 2023,     # arXiv comment "NeurIPS 2023 camera-ready"
    "rando2024universal": 2024,        # arXiv comment "conference paper in ICLR 2024". S2 said 2023
    "li2024badedit": 2024,             # arXiv comment "ICLR 2024"
    "cheng2023nlpsurvey": 2025,        # CrossRef: TNNLS vol. 36, issued 2025-08. S2 said 2023
    "bowen2024jailbreaktuning": 2025,  # CrossRef: AAAI vol. 39 (AAAI-25), issued 2025-04. S2 said 2024
    "qi2024finetuning": 2024,          # iclr.cc/virtual/2024/oral/19735. S2 said 2023
    "carlini2022contrastive": 2022,    # iclr.cc ICLR 2022 orals/posters list. S2 said 2021
    "xu2024shadowcast": 2024,          # arXiv comment "NeurIPS 2024"
    "lyu2024trojvlm": 2024,            # arXiv comment "ECCV 2024"
    "lyu2025vlood": 2025,              # arXiv comment "ICLR 2025". S2 said 2024
    "agarwal2024gkd": 2024,            # arXiv comment "Accepted at ICLR 2024". S2 said 2023
    "kirkpatrick2017ewc": 2017,        # CrossRef: PNAS vol. 114, issued 2017-03. S2 said 2016
    "li2017lwf": 2018,                 # TPAMI vol. 40 (2018), pp. 2935-2947 per S2 journal field;
                                       # arXiv comment "updated with journal version". S2 said 2016
    "ouyang2022instructgpt": 2022,     # NeurIPS 2022: DOI prefix 10.52202/068431, same volume as souri2022
    "luo2023forgetting": 2025,         # CrossRef: IEEE/ACM TASLP vol. 33, 2025. S2 said 2023
    "biderman2024lora": 2024,          # arXiv comment "accepted to TMLR, August 2024"
    "liu2018finepruning": 2018,        # CrossRef: RAID 2018, LNCS, issued 2018
    "tran2018spectral": 2018,          # arXiv comment "accepted to NIPS 2018"
    "chen2018activation": 2019,        # ceur-ws.org/Vol-2301: SafeAI 2019. S2 said 2018
    "azizi2021tminer": 2021,           # arXiv comment "Accepted to Usenix Security 2021"
    "zeng2024beear": 2024,             # CrossRef: EMNLP 2024, DOI 10.18653/v1/2024.emnlp-main.732
    "adi2018watermark": 2018,          # usenix.org/conference/usenixsecurity18/presentation/adi
    "xu2024fingerprint": 2024,         # arXiv comment "Accepted at NAACL 2024"
    "carlini2019secretsharer": 2019,   # usenix.org/conference/usenixsecurity19/presentation/carlini. S2 said 2018
    "yang2025gateddeltanet": 2025,     # arXiv comment "ICLR 2025 camera ready". S2 said 2024
    "penedo2024fineweb": 2024,         # CrossRef: NeurIPS 37 (2024), DOI 10.52202/079017-0970
    "wilson1927": 1927,                # CrossRef: JASA vol. 22, issued 1927-06
    "bagdasaryan2021blind": 2021,      # usenix.org/conference/usenixsecurity21/presentation/bagdasaryan. S2 said 2020
    "kim2016seqkd": 2016,              # arXiv comment "EMNLP 2016"
    "goh2021multimodal": 2021,         # CrossRef: Distill vol. 6, issued 2021-03
    "radford2021clip": 2021,           # proceedings.mlr.press/v139/radford21a (ICML 2021)
    "li2021nad": 2021,                 # arXiv comment "ICLR 2021"
    "betley2025emergent": 2026,        # CrossRef: Nature vol. 649, issued 2026-01; arXiv comment
                                       # "published in Nature 2026/1". S2 said 2025
})
# Meetings that did not take place in a given year (a booktitle naming one is wrong).
NOT_HELD = {"NAACL": {2017, 2020, 2023}, "ICCV": "even", "ECCV": "odd"}
# Latest month in which each venue meets: an arXiv posting after it cannot be that year's paper.
MEETS_BY = {"ICLR": 5, "ICML": 7, "NeurIPS": 12, "CVPR": 6, "ICCV": 10, "ECCV": 10, "(ACL)": 8,
            "EMNLP": 12, "NAACL": 6, "AAAI": 3, "S\\&P": 5, "USENIX": 8, "CCS": 11, "European Symposium": 9,
            "FOCS": 11, "Artificial Intelligence and Statistics": 5, "NDSS": 2, "Recent Advances in Intrusion": 9,
            "Fairness, Accountability": 6}
# Booktitle overrides where S2's venue drops the track.
VENUE_OVERRIDE = {"huang2024composite": "Findings of the Association for Computational Linguistics: NAACL"}
# Venue overrides for records whose S2 venue is empty; each was checked by DOI.
OVERRIDE = {
    "liu2018trojaning": ("inproceedings", "Network and Distributed System Security Symposium (NDSS)"),
    "wilson1927": ("article", "Journal of the American Statistical Association"),
}
CONF_HINT = ("conference", "symposium", "workshop", "naacl", "safeai", "proceedings")
# DOIs that S2 attaches to a record but that belong to a different version of the paper.
DROP_DOI = {"li2017lwf"}  # 10.1007/978-3-319-46493-0_37 is the ECCV 2016 paper, not TPAMI 2018

def year_evidence(kind, p, doi):
    """Venue years readable from the S2 record itself (not S2 `year`, which is the preprint's)."""
    ev = []
    jn = (p.get("journal") or {}).get("name") or ""
    m = re.search(r"\b(19|20)\d\d\b", jn) if jn.lower() != "arxiv" else None
    if m: ev.append(("S2 journal.name", int(m.group(0))))
    # Proceedings DOIs carry the meeting year (CVPR46437.2021.x, v1/2024.naacl-long.x); journal
    # DOIs carry the early-access year instead, so only proceedings DOIs count.
    if kind == "inproceedings" and doi and not doi.startswith("10.48550"):
        m = re.search(r"[./]((?:19|20)\d\d)\.", doi)
        if m: ev.append(("proceedings DOI", int(m.group(1))))
    return ev

def check_year(key, kind, venue, year, ysrc, ev, arx):
    errs = []
    if kind == "arxiv":
        if arx and re.match(r"\d{4}\.", arx) and year != 2000 + int(arx[:2]):
            errs.append(f"arXiv entry year {year} but arXiv id {arx}")
        return errs
    if ysrc == "S2 year":
        errs.append(f"no venue-year evidence for '{venue}' (S2 year {year} is the preprint year); add YEAR_OVERRIDE")
    for src, y in ev:
        if y != year: errs.append(f"year {year} ({ysrc}) disagrees with {src} {y}")
    for v, bad in NOT_HELD.items():
        if v in venue and (year in bad if isinstance(bad, set) else (year % 2 == 0) == (bad == "even")):
            errs.append(f"{v} was not held in {year}")
    if arx and re.match(r"\d{4}\.", arx):
        py, pm = 2000 + int(arx[:2]), int(arx[2:4])
        if year < py: errs.append(f"venue year {year} precedes arXiv posting {arx}")
        for v, mon in MEETS_BY.items():
            if v in venue and kind == "inproceedings" and (py, pm) > (year, mon):
                errs.append(f"arXiv {arx} was posted after {v.strip('()')} {year} met (month {mon})")
    return errs

YEAR_ERRORS = []

def bib_entry(c, p):
    # authors_arxiv: S2 merged distinct authors of these records into one profile (v0.6), so
    # the list is the arXiv listing's, fetched from export.arxiv.org with the same batch.
    names = c.get("authors_arxiv") or [a["name"] for a in p.get("authors", [])]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup: YEAR_ERRORS.append((c["key"], f"duplicate author names {dup}; supply authors_arxiv"))
    authors = " and ".join(names)
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
    venue = VENUE_OVERRIDE.get(c["key"]) or PRETTY.get(venue, venue)
    if c["key"] in DROP_DOI: doi = None
    ev = year_evidence("inproceedings" if kind == "inproceedings" else kind, p, doi)
    if c["key"] in YEAR_OVERRIDE: year, ysrc = YEAR_OVERRIDE[c["key"]], "YEAR_OVERRIDE"
    elif kind != "arxiv" and ev: year, ysrc = ev[0][1], ev[0][0]
    else: year, ysrc = p.get("year") or "", ("arXiv" if kind == "arxiv" else "S2 year")
    YEAR_ERRORS.extend((c["key"], e) for e in check_year(c["key"], kind, venue, year, ysrc, ev, arx))
    L = [f"@{'article' if kind in ('article','arxiv') else 'inproceedings'}{{{c['key']},",
         f"  title = {{{{{latex_escape(p['title'])}}}}},", f"  author = {{{authors}}},"]
    if kind == "arxiv":
        L.append(f"  journal = {{arXiv preprint arXiv:{arx}}},")
    else:
        L.append(f"  {'journal' if kind == 'article' else 'booktitle'} = {{{venue if '&' not in venue or '\\&' in venue else latex_escape(venue)}}},")
    L.append(f"  year = {{{year}}},")
    if doi and kind != "arxiv": L.append(f"  doi = {{{doi}}},")
    L.append("}")
    return "\n".join(L), venue, arx, doi, year, ysrc, names

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
    e, venue, arx, doi, year, ysrc, names = bib_entry(c, p)
    bib.append(e)
    shown = f"arXiv preprint" if venue.lower().startswith("arxiv") or not venue else venue.replace("\\&", "&")
    rows.append(f"| `{c['key']}` | {names[0] if names else '?'} | {year} | {p.get('year')} | {ysrc} | {shown} | {ident} | {p.get('citationCount','')} |")
if YEAR_ERRORS:
    for k, e in YEAR_ERRORS: print("YEAR/AUTHOR CHECK FAILED", k, "-", e, file=sys.stderr)
    sys.exit(f"{len(YEAR_ERRORS)} venue-year/author check failure(s); refs_auto.bib NOT written")
TEXT_FIX = {"Ma\u0327dry": "M{\\k{a}}dry"}  # S2 encodes the ogonek as a combining cedilla
TEXT_FIX[" : [Extended Abstract]"] = ""  # FOCS proceedings suffix on goldwasser2022undetectable
TEXT_FIX["Mart'in Soto"] = "Mart{\\'i}n Soto"  # S2 writes the acute accent as an apostrophe
TEXT_FIX["Si-Yuan Liang"] = "Siyuan Liang"  # v0.6 refetch split the name; the v0.4 record had it whole
TEXT_FIX["SHIYANG LI"] = "Shiyang Li"  # S2 record of yan2024vpi has the name in capitals
def _fix(t):
    for a, b in TEXT_FIX.items(): t = t.replace(a, b)
    return t
bib = [_fix(e) for e in bib]
(here.parent / "refs_auto.bib").write_text(
    "% AUTO-GENERATED by paper/scripts/build_bib.py from a Semantic Scholar batch lookup.\n"
    "% Every entry was resolved by arXiv id / DOI and title-matched. Do not hand-edit;\n% manual, separately verified entries go in refs_manual.bib.\n\n" + "\n\n".join(bib) + "\n")
md = ["# Reference verification", "", f"Semantic Scholar batch lookup by arXiv id / DOI, title-matched. **{len(rows)} verified, {len(bad)} unverified.**",
      "Generated with refs_auto.bib by `scripts/build_bib.py`; *year* is the year printed in the .bib, *S2 year* is "
      "Semantic Scholar's (the preprint's), and *year source* says where the printed year came from.", "",
      "| key | first author | year | S2 year | year source | venue (as printed) | id | citations |",
      "|---|---|---|---|---|---|---|---|"] + rows
if bad:
    md += ["", "## UNVERIFIED — not in refs_auto.bib", "", "| key | id | reason |", "|---|---|---|"] + [f"| `{k}` | {i} | {w} |" for k, i, w in bad]
(here.parent / "refs_verification.md").write_text("\n".join(md) + "\n")
print(len(rows), "verified;", len(bad), "bad")
for b in bad: print("BAD", *b)
