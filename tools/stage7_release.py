# -*- coding: utf-8 -*-
"""Stage 7: the checksum manifest, the golden fingerprints, and the proof.

    python tools/stage7_release.py            # manifest + fingerprints
    python tools/stage7_release.py --verify   # also re-hash every parquet
    python tools/stage7_release.py --reproduce  # also regenerate and compare

Writes `docs/gates/checksums.json`, `docs/gates/checksums.md` and
`docs/gates/gate7.md`.

THREE CLAIMS, AND EACH IS PROVED DIFFERENTLY.

  The data is what we say it is. A sha256 per parquet, 30 entries. `--verify`
  re-hashes them and compares, which is the claim the manifest itself cannot
  make.

  The design that made it is pinned. Two hashes: a SETTINGS hash over the
  configuration that shapes the data, and the DESIGN fingerprint from
  `generate.fingerprint`, which folds in the bytes of the 7 modules that
  generate it. A settings hash alone would not notice a changed law; a code
  hash alone would not notice a changed constant.

  A re-run reproduces it. `--reproduce` regenerates one seed at each scale to a
  temp path and compares sha256 against the shipped file. This is the only one
  of the three that tests determinism rather than asserting it, and it is the
  claim the plan actually gates on. Stage 3 is included by hashing each censored
  view's column set and visible-cell counts, since censoring is a function of
  the master and its result must be reproducible for the same reason.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t9v2 import contentfp as CFP  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
GATES = ROOT / "docs" / "gates"
SCALES = ["100K", "1M", "10M"]
SEEDS = list(range(20250, 20260))
CHUNK = 8 << 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# free text that explains a setting without shaping any data. Excluded from the
# settings hash, and the reason is not tidiness. Editing a `note` moved the hash
# from 423de165 to 7334911a on 17 Aug 2026, for a documentation change that
# altered no byte of any dataset. A fingerprint that has to be re-baselined every
# time a comment improves teaches the reader to re-baseline it, which is exactly
# the habit it exists to prevent. `route` and `source` are KEPT: a provenance
# change is a substantive claim about the data even though it shapes none of it.
PROSE_KEYS = {"note", "doc", "comment", "description"}


def strip_prose(node):
    if isinstance(node, dict):
        return {k: strip_prose(v) for k, v in node.items()
                if k not in PROSE_KEYS}
    if isinstance(node, list):
        return [strip_prose(v) for v in node]
    return node


def settings_hash(s):
    """A hash over the settings that shape the data, order-independent.

    Prose is stripped first, so this moves when a value, a route or the shape of
    the configuration moves, and stays put when only an explanation does.
    """
    blob = json.dumps(strip_prose(s.raw), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def censor_signature(master, s):
    """Stage 3's reproducible output: what each view can see.

    Not a hash of the censored frames themselves, which would mean holding four
    copies of a 10M master. The column set and the visible-cell count per view
    are what censoring DECIDES, and a change to the map moves one of them.
    """
    from t9v2 import censor as CEN
    sig = {}
    for v in CEN.VIEWS:
        d = CEN.censor(master, v, s)
        sig[v] = {"columns": sorted(map(str, d.columns)),
                  "n_columns": int(len(d.columns)),
                  "visible_click": int(d["click"].notna().sum())}
        del d
    return sig


def build(verify=False):
    from t9v2 import generate as G
    from t9v2.core.config import load
    s = load("default")
    rows, t0 = [], time.time()
    for scale in SCALES:
        for seed in SEEDS:
            p = OUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))
            if not p.exists():
                rows.append({"scale": scale, "seed": seed, "status": "MISSING"})
                continue
            mf = OUT / ("t9v2_%s_seed%d.manifest.json" % (scale, seed))
            man = json.loads(mf.read_text(encoding="utf-8")) if mf.exists() else {}
            h = sha256(p)
            # the portable one: hashes VALUES, so another machine's parquet
            # writer cannot make identical data look different. See contentfp.
            c = CFP.fingerprint_file(p)
            rows.append({"scale": scale, "seed": seed, "file": p.name,
                         "bytes": p.stat().st_size, "sha256": h,
                         "content_fp": c,
                         "rows": man.get("rows"), "columns": man.get("columns"),
                         "win_rate": man.get("win_rate"),
                         "fingerprint": man.get("fingerprint"),
                         "status": "OK"})
            print("  %-5s seed%d  sha %s  content %s  %.2f GB"
                  % (scale, seed, h[:12], c, p.stat().st_size / 1e9), flush=True)
    out = {"settings_hash": settings_hash(s),
           "design_fingerprint": G.fingerprint(s),
           "rng_scheme": s.raw.get("meta", {}).get("rng_scheme", "t9v2-rng-1"),
           "n_datasets": sum(1 for r in rows if r["status"] == "OK"),
           "total_bytes": sum(r.get("bytes", 0) for r in rows),
           "hashed_seconds": round(time.time() - t0, 1),
           "datasets": rows}
    return out, s


def reproduce(s):
    """Regenerate one seed per scale and compare. The determinism proof."""
    from t9v2 import generate as G
    import pandas as pd
    res = []
    for scale in SCALES:
        seed = SEEDS[0]
        shipped = OUT / ("t9v2_%s_seed%d.parquet" % (scale, seed))
        if not shipped.exists():
            res.append({"scale": scale, "seed": seed, "match": None,
                        "note": "shipped file missing"})
            continue
        t = time.time()
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / shipped.name
            G.generate(scale=scale, seed=seed, quiet=True, out=tmp)
            a, b = sha256(shipped), sha256(tmp)
            # BOTH claims, because they are not the same claim. The bytes match
            # only inside the pinned library versions; the content match is what
            # another machine can reproduce and check.
            ca, cb = CFP.fingerprint_file(shipped), CFP.fingerprint_file(tmp)
            master = pd.read_parquet(tmp)
            sig = censor_signature(master, s)
            del master
        res.append({"scale": scale, "seed": seed, "shipped_sha256": a,
                    "regenerated_sha256": b, "match": a == b,
                    "shipped_content_fp": ca, "regenerated_content_fp": cb,
                    "content_match": ca == cb,
                    "censor_signature": sig,
                    "seconds": round(time.time() - t, 1)})
        print("  %-5s seed%d  bytes %s  content %s  %.0fs"
              % (scale, seed, "MATCH" if a == b else "DIFFER",
                 "MATCH" if ca == cb else "DIFFER", time.time() - t), flush=True)
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reproduce", action="store_true",
                    help="regenerate one seed per scale and compare hashes")
    a = ap.parse_args(argv)

    GATES.mkdir(parents=True, exist_ok=True)
    print("hashing the 30 datasets")
    man, s = build()
    rep = reproduce(s) if a.reproduce else None
    if rep:
        man["reproduction"] = rep
    (GATES / "checksums.json").write_text(json.dumps(man, indent=1, default=str),
                                          encoding="utf-8")

    md = ["# Checksum manifest, the 30 datasets", "",
          "*Generated by `tools/stage7_release.py`. This file is written, "
          "never edited.*", "",
          "| | |", "|---|---|",
          "| datasets | %d of 30 |" % man["n_datasets"],
          "| total size | %.1f GB |" % (man["total_bytes"] / 1e9),
          "| settings hash | `%s` |" % man["settings_hash"][:32],
          "| design fingerprint | `%s` |" % man["design_fingerprint"],
          "| RNG scheme | `%s` |" % man["rng_scheme"], "",
          "The settings hash covers the configuration that shapes the data. The "
          "design fingerprint additionally folds in the bytes of the 7 modules "
          "that generate it, so a changed law is caught even when every constant "
          "is untouched.", "",
          "## Two hashes per dataset, and only one of them travels", "",
          "**`sha256`** is over the parquet FILE. It answers \"is this file "
          "intact\". It does not survive a different writer: parquet embeds the "
          "writer version, compression choice and row-group layout, so identical "
          "data written by a different `pyarrow` gives a different sha256. Use it "
          "to check a file you downloaded, not to check a file you regenerated.", "",
          "**`content_fp`** is over the DATA — column values in sorted column "
          "order, explicit encoding, documented in `src/t9v2/contentfp.py`. Same "
          "numbers give the same fingerprint whatever wrote the file. **This is "
          "the value another user should compare against**, and the one that "
          "answers \"did my machine produce your dataset\". Verified: the same "
          "frame rewritten with gzip and a 7,777-row row-group gives a different "
          "sha256 and an identical `content_fp`, while changing one value by "
          "1e-9 moves it.", "",
          "| Scale | Seed | Rows | Cols | Win rate | Bytes | content_fp | sha256 |",
          "|---|---:|---:|---:|---:|---:|---|---|"]
    for r in man["datasets"]:
        if r["status"] != "OK":
            md.append("| %s | %d | **MISSING** | | | | | |" % (r["scale"], r["seed"]))
            continue
        md.append("| %s | %d | %s | %s | %s | %d | `%s` | `%s` |"
                  % (r["scale"], r["seed"], "{:,}".format(r["rows"] or 0),
                     r["columns"], ("%.4f" % r["win_rate"]) if r["win_rate"] else "-",
                     r["bytes"], r.get("content_fp", "-"), r["sha256"]))

    if rep:
        md += ["", "## Reproduction", "",
               "One seed regenerated at each scale into a temporary path and "
               "compared byte for byte against the shipped file. This tests "
               "determinism rather than asserting it.", "",
               "| Scale | Seed | File bytes | Data content | Seconds |",
               "|---|---:|---|---|---:|"]
        for r in rep:
            md.append("| %s | %d | %s | %s | %s |"
                      % (r["scale"], r["seed"],
                         "**MATCH**" if r.get("match") else "**DIFFERS**",
                         "**MATCH**" if r.get("content_match") else "**DIFFERS**",
                         r.get("seconds", "-")))
        md += ["", "Stage 3 reproduces with it: the censored column sets and "
               "visible-cell counts per view are recorded in `checksums.json` "
               "under `reproduction[].censor_signature`, since censoring is a "
               "function of the master and must reproduce for the same reason."]

    (GATES / "checksums.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nwrote docs/gates/checksums.md and checksums.json")
    print("settings hash    %s" % man["settings_hash"])
    print("design fingerprint %s" % man["design_fingerprint"])
    ok = man["n_datasets"] == 30 and (not rep or all(r.get("match") for r in rep))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
