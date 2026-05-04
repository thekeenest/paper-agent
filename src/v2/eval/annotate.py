"""
Interactive annotation CLI for PaperAffilBench.

Produces gold/<doi_safe>.json files with author-affiliation ground truth
in the format expected by metrics.py.

Gold JSON schema
----------------
::

    {
      "paper_id": "arxiv:1901.04341",
      "doi": "10.5555/335791284",
      "title": "...",
      "year": 2023,
      "venue": "NeurIPS",
      "annotator": "annotator_a",
      "annotations": [
        {
          "author_name": "Alice Smith",
          "position": 0,
          "normalized_aff": "Massachusetts Institute of Technology",
          "ror_id": "https://ror.org/042nb2s44",
          "country_code": "US",
          "type": "education",
          "evidence_span": "A. Smith (MIT)",
          "email": "alice@mit.edu"
        }
      ]
    }

Two-annotator workflow
----------------------
Annotator A writes ``gold/<doi>.json``; annotator B writes ``gold/<doi>.b.json``.
Running ``annotate.py merge --doi <doi>`` produces a reconciled
``gold/<doi>.json`` from both, flagging disagreements.

Usage
-----
::

    python -m src.v2.eval.annotate annotate --doi 10.5555/335791284 --annotator a
    python -m src.v2.eval.annotate annotate --doi 10.5555/335791284 --annotator b
    python -m src.v2.eval.annotate merge --doi 10.5555/335791284
    python -m src.v2.eval.annotate status
    python -m src.v2.eval.annotate validate --doi 10.5555/335791284
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

_MANIFEST_PATH = Path(
    os.getenv(
        "BENCH_MANIFEST",
        "benchmark/PaperAffilBench/manifest.json",
    )
)
_GOLD_DIR = Path(os.getenv("BENCH_GOLD_DIR", "benchmark/PaperAffilBench/gold"))

_ANNOTATION_FIELDS = [
    "author_name",
    "position",
    "normalized_aff",
    "ror_id",
    "country_code",
    "type",
    "evidence_span",
    "email",
]

_ORG_TYPES = ("education", "company", "facility", "government", "nonprofit", "other")


# ─────────────────────────── helpers ─────────────────────────────────────────


def _load_manifest() -> dict[str, Any]:
    if not _MANIFEST_PATH.exists():
        print(f"Error: manifest not found at {_MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(_MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _doi_safe(doi: str) -> str:
    """Convert DOI to a filename-safe string."""
    return doi.replace("/", "_").replace(".", "_")


def _gold_path(doi: str, annotator: str = "") -> Path:
    safe = _doi_safe(doi)
    suffix = f".{annotator}.json" if annotator and annotator != "a" else ".json"
    return _GOLD_DIR / f"{safe}{suffix}"


def _paper_by_doi(manifest: dict[str, Any], doi: str) -> dict[str, Any] | None:
    for p in manifest.get("papers", []):
        if p.get("doi") == doi or p.get("paper_id") == doi:
            return p
    return None


def _prompt(label: str, default: str = "") -> str:
    """Read a line from stdin, with optional default."""
    hint = f" [{default}]" if default else ""
    try:
        val = input(f"  {label}{hint}: ").strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        sys.exit(0)


def _prompt_int(label: str, default: int = 0) -> int:
    val = _prompt(label, str(default))
    try:
        return int(val)
    except ValueError:
        return default


# ─────────────────────────── sub-commands ────────────────────────────────────


def _cmd_annotate(args: argparse.Namespace) -> None:
    """Interactive annotation session for a single paper."""
    manifest = _load_manifest()
    doi = args.doi
    paper = _paper_by_doi(manifest, doi)
    if not paper:
        print(f"Error: paper with DOI {doi!r} not found in manifest.", file=sys.stderr)
        sys.exit(1)

    annotator = args.annotator or "a"
    out_path = _gold_path(doi, annotator)

    if out_path.exists() and not args.force:
        print(f"Annotation already exists: {out_path}")
        print("Use --force to overwrite.")
        return

    _GOLD_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Annotating: {paper['title']}")
    print(f"  DOI : {doi}")
    print(f"  Year: {paper['year']}  Venue: {paper['venue']}")
    print(f"  n_authors: {paper.get('n_authors', '?')}")
    print(f"  Annotator: {annotator}")
    print(f"{'='*60}\n")

    # Load existing annotations to allow resumption
    existing: list[dict[str, Any]] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())["annotations"]
            print(f"Resuming from {len(existing)} existing annotations.")
        except Exception:
            existing = []

    n_authors = int(paper.get("n_authors", 3))
    annotations: list[dict[str, Any]] = list(existing)

    print(f"Enter data for each of the {n_authors} author(s). Leave blank to skip a field.\n")
    print("Org type options:", ", ".join(_ORG_TYPES))
    print("(Press Ctrl+C at any point to save and exit.)\n")

    start_idx = len(annotations)
    for pos in range(start_idx, n_authors):
        print(f"--- Author {pos + 1} of {n_authors} ---")
        ann: dict[str, Any] = {
            "author_name": _prompt("author_name"),
            "position": pos,
            "normalized_aff": _prompt("normalized_aff (full institution name)"),
            "ror_id": _prompt("ror_id (https://ror.org/... or blank)"),
            "country_code": _prompt("country_code (ISO 3166-1 alpha-2, e.g. US)"),
            "type": _prompt("org type", default="education"),
            "evidence_span": _prompt("evidence_span (text snippet from PDF)"),
            "email": _prompt("email (if found, else blank)"),
        }
        # Normalize
        if ann["type"] not in _ORG_TYPES:
            print(f"  Warning: unknown org type {ann['type']!r}, using 'other'")
            ann["type"] = "other"
        ann["country_code"] = ann["country_code"].upper()[:2]
        annotations.append(ann)
        print()

    gold: dict[str, Any] = {
        "paper_id": paper.get("paper_id", ""),
        "doi": doi,
        "title": paper.get("title", ""),
        "year": paper.get("year"),
        "venue": paper.get("venue", "NeurIPS"),
        "annotator": annotator,
        "annotations": annotations,
    }
    out_path.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {len(annotations)} annotations → {out_path}")


def _cmd_merge(args: argparse.Namespace) -> None:
    """Reconcile annotator A and B files for a DOI."""
    doi = args.doi
    path_a = _gold_path(doi, "a")
    path_b = _gold_path(doi, "b")
    out_path = _gold_path(doi, "")  # same as 'a' path — final gold

    if not path_a.exists():
        print(f"Error: annotator A file not found: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.exists():
        print(f"Error: annotator B file not found: {path_b}", file=sys.stderr)
        sys.exit(1)

    gold_a: dict[str, Any] = json.loads(path_a.read_text())
    gold_b: dict[str, Any] = json.loads(path_b.read_text())

    anns_a = gold_a.get("annotations", [])
    anns_b = gold_b.get("annotations", [])

    merged: list[dict[str, Any]] = []
    disagreements: list[str] = []

    n = max(len(anns_a), len(anns_b))
    for i in range(n):
        ann_a = anns_a[i] if i < len(anns_a) else None
        ann_b = anns_b[i] if i < len(anns_b) else None

        if ann_a is None:
            merged.append(ann_b)  # type: ignore[arg-type]
            continue
        if ann_b is None:
            merged.append(ann_a)
            continue

        # Merge: prefer non-empty values; flag disagreements
        merged_ann: dict[str, Any] = {}
        for field in _ANNOTATION_FIELDS:
            va = ann_a.get(field, "")
            vb = ann_b.get(field, "")
            if va == vb:
                merged_ann[field] = va
            elif not va:
                merged_ann[field] = vb
            elif not vb:
                merged_ann[field] = va
            else:
                # Real disagreement — use A's value, flag it
                merged_ann[field] = va
                disagreements.append(
                    f"Position {i}, field {field!r}: A={va!r} vs B={vb!r}"
                )
        merged.append(merged_ann)

    final: dict[str, Any] = {
        **gold_a,
        "annotator": "merged",
        "annotations": merged,
        "merge_disagreements": disagreements,
    }
    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Merged {n} annotations → {out_path}")
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s) flagged:")
        for d in disagreements[:10]:
            print(f"  {d}")
        if len(disagreements) > 10:
            print(f"  ... ({len(disagreements) - 10} more)")
    else:
        print("No disagreements — perfect annotator agreement.")


def _cmd_status(_args: argparse.Namespace) -> None:
    """Show annotation progress across the manifest."""
    manifest = _load_manifest()
    papers = manifest.get("papers", [])
    _GOLD_DIR.mkdir(parents=True, exist_ok=True)

    done: list[str] = []
    partial: list[str] = []  # one annotator only
    missing: list[str] = []

    for p in papers:
        doi = p["doi"]
        path_a = _gold_path(doi, "a")
        path_b = _gold_path(doi, "b")
        path_merged = _gold_path(doi, "")

        if path_merged.exists() and path_a != path_merged:
            done.append(doi)
        elif path_a.exists() and path_b.exists():
            done.append(doi)
        elif path_a.exists() or path_b.exists():
            partial.append(doi)
        else:
            missing.append(doi)

    total = len(papers)
    print(f"\nPaperAffilBench annotation status ({total} papers total)")
    print(f"  {'Done (both annotators)':35s}: {len(done):4d}  ({100*len(done)/total:.1f}%)")
    print(f"  {'Partial (one annotator)':35s}: {len(partial):4d}  ({100*len(partial)/total:.1f}%)")
    print(f"  {'Missing':35s}: {len(missing):4d}  ({100*len(missing)/total:.1f}%)")

    by_year: dict[int, dict[str, int]] = {}
    for p in papers:
        yr = p["year"]
        if yr not in by_year:
            by_year[yr] = {"done": 0, "partial": 0, "missing": 0}
        doi = p["doi"]
        if doi in done:
            by_year[yr]["done"] += 1
        elif doi in partial:
            by_year[yr]["partial"] += 1
        else:
            by_year[yr]["missing"] += 1

    print("\n  By year:")
    for yr in sorted(by_year):
        s = by_year[yr]
        print(f"    {yr}: done={s['done']} partial={s['partial']} missing={s['missing']}")

    if missing:
        print(f"\n  Next unannotated: {missing[0]}")


def _cmd_validate(args: argparse.Namespace) -> None:
    """Validate the gold JSON for a DOI."""
    doi = args.doi
    path = _gold_path(doi, "")
    if not path.exists():
        path = _gold_path(doi, "a")
    if not path.exists():
        print(f"No gold file found for {doi}", file=sys.stderr)
        sys.exit(1)

    gold: dict[str, Any] = json.loads(path.read_text())
    errors: list[str] = []

    if not gold.get("doi"):
        errors.append("Missing 'doi' field")
    if not gold.get("annotations"):
        errors.append("Empty 'annotations' list")

    for i, ann in enumerate(gold.get("annotations", [])):
        if not ann.get("author_name"):
            errors.append(f"annotations[{i}]: missing author_name")
        if ann.get("type") and ann["type"] not in _ORG_TYPES:
            errors.append(f"annotations[{i}]: invalid type {ann['type']!r}")
        ror = ann.get("ror_id", "")
        if ror and not ror.startswith("https://ror.org/"):
            errors.append(f"annotations[{i}]: ror_id must start with https://ror.org/")

    if errors:
        print(f"Validation FAILED for {doi} ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        n = len(gold["annotations"])
        print(f"OK: {doi} — {n} annotation(s) valid")


# ─────────────────────────── CLI ─────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.v2.eval.annotate",
        description="Interactive annotation CLI for PaperAffilBench.",
    )
    sub = parser.add_subparsers(dest="command")

    # annotate
    p_ann = sub.add_parser("annotate", help="Annotate a single paper interactively")
    p_ann.add_argument("--doi", required=True, help="Paper DOI (e.g. 10.5555/335791284)")
    p_ann.add_argument(
        "--annotator",
        default="a",
        choices=["a", "b"],
        help="Annotator ID (a or b, default a)",
    )
    p_ann.add_argument("--force", action="store_true", help="Overwrite existing annotation")
    p_ann.set_defaults(func=_cmd_annotate)

    # merge
    p_merge = sub.add_parser("merge", help="Reconcile annotator A and B files")
    p_merge.add_argument("--doi", required=True)
    p_merge.set_defaults(func=_cmd_merge)

    # status
    p_status = sub.add_parser("status", help="Show annotation progress")
    p_status.set_defaults(func=_cmd_status)

    # validate
    p_validate = sub.add_parser("validate", help="Validate a gold JSON file")
    p_validate.add_argument("--doi", required=True)
    p_validate.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
