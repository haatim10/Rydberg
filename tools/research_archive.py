"""Inventory and verify the 2026-10-30 research upload without changing experiments.

Uses only the standard library. No experiments, model deserialization, or network
requests are performed. Run from any directory: python tools/research_archive.py
inventory, then python tools/research_archive.py verify [--local-only].
"""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import csv
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/archive_20261030"
BASE = "0162cf515bf34354fcf151b9adbdbe7cde887711"
BRANCHES = {
    "foundation": "research/self-supervised-20261030",
    "receiver": "research/tcom-receiver-20261030",
    "comparison": "research/urformer-comparison-20261030",
    "audits": "research/urformer-audits-20261030",
    "paper": "paper/merged-tcom-20261030",
}
FIELDS = ["path", "bytes", "sha256", "section", "reason"]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def section(path):
    parts = Path(path).parts
    if parts[0] in ("paper", "output"):
        return "paper"
    if parts[0] == "experiments":
        return "audits"
    name = parts[1] if parts[0] == "results" and len(parts) > 1 else parts[0]
    if name.startswith(("matched_supervision", "urformer_reproduction_", "receiver_urformer_", "tcom_review_checks_")):
        return "comparison"
    if name.startswith("tcom_"):
        return "receiver"
    return "foundation"


def epoch_weights(paths):
    """Preserve last saved and explicitly selected epochs, not every epoch."""
    groups = defaultdict(dict)
    selected = {}
    for name in paths:
        p = Path(name)
        match = re.fullmatch(r"epoch_(\d+)\.pt", p.name)
        if match:
            groups[p.parent][int(match[1])] = name
    for parent, epochs in groups.items():
        selected[epochs[max(epochs)]] = "last_saved_epoch_weight"
        result = ROOT / parent / "result.json"
        if result.exists():
            obj = json.loads(result.read_text())
            epoch = obj.get("chosen_epoch_1based", obj.get("chosen_epoch"))
            if isinstance(epoch, int) and epoch in epochs:
                selected[epochs[epoch]] = "validation_selected_epoch_weight"
    # This later frozen selection may differ from a run's initial result.json.
    p = ROOT / "results/paper2_hankel_confirmation/selection.json"
    if p.exists():
        for mode, data in json.loads(p.read_text())["modes"].items():
            for key, epoch in data["epochs"].items():
                seed, arm = key.split("/")
                name = f"results/paper2_hankel_confirmation/{mode}/seed{seed}/development/{arm}/epoch_{epoch:03d}.pt"
                if name not in paths:
                    raise ValueError(f"Missing selected epoch: {name}")
                selected[name] = "frozen_confirmation_selected_weight"
    return selected


def classify(name, size, chosen):
    p = Path(name)
    if p.parts[0] == "tmp":
        return False, "temporary_extraction_or_render"
    if any(x in p.parts for x in ("__pycache__", ".pytest_cache", "matplotlib_cache")) or p.name == ".DS_Store":
        return False, "cache_or_os_metadata"
    if p.suffix in (".aux", ".out", ".blg", ".lock") or name == "texput.log" or (p.parts[0] == "paper" and p.suffix == ".log"):
        return False, "build_or_process_artifact"
    if "reference" in p.parts and p.suffix in (".pdf", ".html", ".png", ".jpg", ".jpeg", ".txt"):
        return False, "third_party_reference_copy_link_original_instead"
    if name.startswith("results/paper2_reproduction/source/"):
        return False, "third_party_reference_copy_link_original_instead"
    if name in chosen:
        return True, chosen[name]
    if p.suffix == ".pt":
        if any(x in p.parts for x in ("data", "training_data")):
            return False, "generated_input_data"
        if any(token in p.stem for token in ("best", "final", "selected", "constant_control")):
            return True, "selected_or_final_weight"
        return False, "intermediate_or_resume_weight"
    if p.suffix in (".npy", ".npz"):
        if any(x in p.parts for x in ("data", "training_data", "predictions", "anchors")):
            return False, "generated_input_or_prediction_bank"
        if size <= 1024 * 1024 and (p.parts[0] == "experiments" or p.name == "val_per_trial.npy"):
            return True, "compact_numeric_evidence"
        return False, "raw_numeric_array"
    if p.suffix in (".zip", ".tar"):
        if "overleaf" in p.name:
            return True, "overleaf_source_package"
        return False, "duplicate_reproducibility_bundle"
    if size >= 50 * 1024 * 1024:
        return False, "large_artifact_requires_separate_storage"
    return True, "code_report_figure_or_protocol"


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def inventory():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "included_files.csv").exists():
        raise SystemExit("Inventory already exists; preserve this dated snapshot.")
    tracked = set(git("ls-tree", "-r", "--name-only", BASE).decode().splitlines())
    paths = sorted(x for x in git("ls-files", "--others", "--exclude-standard", "-z").decode().split("\0") if x)
    paths = [p for p in paths if not p.startswith(("docs/archive_20261030/", "docs/sections/", "tools/")) and p not in ("README.md", "docs/README.md", "results/README.md", "experiments/README.md", "output/README.md")]
    paths = [p for p in paths if (ROOT / p).is_file()]
    chosen = epoch_weights(paths)
    included, excluded = [], []
    for name in paths:
        p = ROOT / name
        if p.is_symlink():
            raise ValueError(f"Review symlink before publishing: {name}")
        size = p.stat().st_size
        keep, reason = classify(name, size, chosen)
        # Hash scientific files, including omitted data. Temporary duplicate
        # extractions and OS/cache files only receive a path/size inventory.
        checksum = "" if reason in ("temporary_extraction_or_render", "cache_or_os_metadata", "build_or_process_artifact") else digest(p)
        row = dict(path=name, bytes=size, sha256=checksum, section=section(name), reason=reason)
        (included if keep else excluded).append(row)
    write_csv(OUT / "included_files.csv", included)
    write_csv(OUT / "local_only_files.csv", excluded)
    rows = []
    # Existing tracked research is already in the base commit. Hash it to check
    # that archive preparation did not change it. paper/README.md is navigation.
    for name in sorted(tracked - {".gitignore", "paper/README.md"}):
        p = ROOT / name
        if p.is_file():
            rows.append(dict(path=name, bytes=p.stat().st_size, sha256=digest(p), section="existing_base", reason="previously_tracked"))
    write_csv(OUT / "existing_files.csv", rows)
    sections = {}
    for key in BRANCHES:
        part = [r for r in included if r["section"] == key]
        sections[key] = dict(branch=BRANCHES[key], files=len(part), bytes=sum(r["bytes"] for r in part))
    reasons = Counter(r["reason"] for r in excluded)
    summary = dict(date="2026-10-30", base_commit=BASE, complete_branch="codex", branches_are_cumulative=True,
        included_files=len(included), included_bytes=sum(r["bytes"] for r in included),
        local_only_files=len(excluded), local_only_bytes=sum(r["bytes"] for r in excluded),
        previously_tracked_files=len(rows), sections=sections, exclusions_by_reason=dict(reasons),
        scope="New previously untracked files visible under the pre-existing .gitignore; existing ignored historic artifacts are not added.",
        user_choice="Upload code, papers, reports, figures and selected/final weights; document the large local-only files.")
    (OUT / "inventory_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


def verify(local_only=False):
    inventories = ["included_files.csv", "existing_files.csv"]
    if local_only:
        inventories.append("local_only_files.csv")
    count = 0
    for inventory_name in inventories:
        with (OUT / inventory_name).open() as f:
            for row in csv.DictReader(f):
                p = ROOT / row["path"]
                if not p.is_file() or p.stat().st_size != int(row["bytes"]):
                    raise SystemExit(f"Missing or changed size: {row['path']}")
                if row["sha256"] and digest(p) != row["sha256"]:
                    raise SystemExit(f"Changed content: {row['path']}")
                count += 1
    print(json.dumps(dict(passed=True, files_checked=count, included_local_only=local_only)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("inventory", "verify"))
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args()
    inventory() if args.action == "inventory" else verify(args.local_only)
