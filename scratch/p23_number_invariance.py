"""PROMPT 23 standing rule: no number may change.

Extracts every numeric literal from a .tex file and compares the multiset
against the same file at a baseline commit. A rewrite may move numbers around,
re-order them, or change the prose that surrounds them, but the SET of values
with their multiplicities must be identical.

Comments are stripped before extraction, so a value that lives only in a
`% src:` provenance comment is not counted as manuscript content -- but the
src comments themselves are checked separately for survival.

Run:  python3 scratch/p23_number_invariance.py <baseline-rev>
"""
from __future__ import annotations

import re
import subprocess
import sys
from collections import Counter

FILES = [
    "paper/thesisproj2/haatim_thesisproj2.tex",
    "wip/spl2/haatim_structural_priors_evaluation.tex",
]

# Values that may legitimately appear that were not at the baseline. Each is
# declared here with its reason so the check stays meaningful rather than being
# quietly widened. NONE of these is a measurement.
#
# PROMPT 23 C2 requires a worked arithmetic example showing why low-SNR samples
# dominate an average ("if one sample has normalized error 2 and another has
# 0.02"). Carrying that arithmetic through produces intermediate values that
# are illustrative, not measured.
ILLUSTRATIVE = {
    0.01: "worked example: the clean sample's error after a halving",
    0.51: "worked example: the average after improving the noisy sample",
    1.01: "worked example: the average of errors 2 and 0.02",
    1.005: "worked example: the average after halving the clean sample's error",
}

# Values CITED from prior work, not measured here. PROMPT 23 A3 requires stating
# the size of the prior work's deweighting gain alongside our own, so that the
# contrast is not read as a like-for-like comparison. Supplied by the brief from
# a human reading of arXiv:2210.14103 v3.
CITED = {
    0.54: "Wiesmayr et al. deweighting gain at 1% BLER (their Section 4)",
}

# A value at the baseline that was WRONG and is corrected here. This is the one
# category that overrides the standing rule, and it is deliberately narrow: the
# correction must be verifiable against a committed scoring artefact, and the
# reason is recorded here rather than left to a commit message.
CORRECTED = {
    -0.704: (-0.703,
             "balanced-loss three-seed mean, [-5,0) dB bin. Exact mean from "
             "reports/p20/p30_score.json is -0.70346, and "
             "reports/p20/PART_B_C3.md table B3.4 already carried -0.703. "
             "-0.704 was mis-transcribed via reports/p22/review_response.md "
             "into the manuscript in the previous turn. Both now fixed."),
}

# A number: optional sign, digits, optional thousands separators ({,} or ,),
# optional decimal part. Captures 1,586,900 / 80{,}000 / 89.7 / -0.177 / 2.171
NUM = re.compile(r"[-+−]?\d[\d,]*(?:\{,\})?[\d,]*(?:\.\d+)?")


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        # a % not preceded by a backslash starts a comment
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def numbers(text: str) -> Counter:
    body = strip_comments(text)
    # normalise thousands separators and unicode minus so 80{,}000 == 80000
    body = body.replace("{,}", "").replace("−", "-")
    vals = []
    for tok in NUM.findall(body):
        tok = tok.replace(",", "").rstrip(".")
        if tok in ("", "-", "+"):
            continue
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return Counter(vals)


def src_comments(text: str) -> Counter:
    return Counter(l.strip() for l in text.split("\n") if l.strip().startswith("% src:"))


def at_rev(rev: str, path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"{rev}:{path}"],
                                       text=True, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None


def main() -> int:
    rev = sys.argv[1] if len(sys.argv) > 1 else "f76ed63"
    bad = False
    for path in FILES:
        old = at_rev(rev, path)
        if old is None:
            print(f"  {path}: not present at {rev} -- skipped")
            continue
        new = open(path, encoding="utf-8").read()
        a, b = numbers(old), numbers(new)
        sa, sb = src_comments(old), src_comments(new)
        lost_src = sa - sb

        # The rule is that no VALUE may change. PROMPT 23 C4 also requires the
        # abstract to stop quoting numbers that the body already carries, which
        # lowers a value's multiplicity without altering it. So a value
        # DISAPPEARING from the document is a violation; a value appearing
        # fewer times while still present is reported but is not.
        lost = set(a) - set(b)
        new_vals = set(b) - set(a)
        fixed = {v: CORRECTED[v] for v in lost if v in CORRECTED}
        lost = lost - set(fixed)
        new_vals = new_vals - {t[0] for t in fixed.values()}
        if fixed:
            print("  CORRECTED values (baseline was wrong; verified):")
            for v, (nv, why) in sorted(fixed.items()):
                print(f"    {v!r} -> {nv!r}")
                print(f"        {why}")
        fewer = {v: (a[v], b[v]) for v in set(a) & set(b) if a[v] != b[v]}

        print(f"\n{path}")
        print(f"  distinct values at {rev}: {len(a)}   now: {len(b)}")
        if lost:
            bad = True
            print("  VALUES LOST ENTIRELY  (violation):")
            for v in sorted(lost):
                print(f"    {v!r} (appeared {a[v]}x)")
        # A value may also be legitimately PORTED between Paper 2's two files
        # (PROMPT 23 Part B). That is only allowed if the value was already in
        # the OTHER file at the baseline -- verified here rather than trusted.
        other = [f for f in FILES if f != path]
        baseline_elsewhere = set()
        for o in other:
            txt = at_rev(rev, o)
            if txt:
                baseline_elsewhere |= set(numbers(txt))
        # the corrected value counts as present wherever its wrong form was
        baseline_elsewhere |= {nv for _, (nv, _) in CORRECTED.items()}
        ported = {v for v in new_vals if v in baseline_elsewhere}
        if ported:
            print("  PORTED from the other Paper 2 file (verified at baseline):")
            for v in sorted(ported):
                print(f"    {v!r}")
        new_vals = new_vals - ported

        declared = {v for v in new_vals if v in ILLUSTRATIVE or v in CITED}
        undeclared = new_vals - declared
        if declared:
            print("  new values, DECLARED (not our measurements):")
            for v in sorted(declared):
                why = ILLUSTRATIVE.get(v) or CITED[v]
                kind = "illustrative" if v in ILLUSTRATIVE else "cited prior work"
                print(f"    {v!r} [{kind}] — {why}")
        if undeclared:
            bad = True
            print("  NEW VALUES not at baseline and not declared  (violation):")
            for v in sorted(undeclared):
                print(f"    {v!r} (appears {b[v]}x)")
        new_vals = undeclared
        if not lost and not new_vals:
            print("  values: IDENTICAL set — no value altered, none lost")
        if fewer:
            print("  multiplicity changes (allowed; value still present):")
            for v, (x, y) in sorted(fewer.items()):
                print(f"    {v!r}: {x} -> {y}")
        print(f"  % src: comments at {rev}: {sum(sa.values())}   now: {sum(sb.values())}")
        if lost_src:
            bad = True
            print("  LOST src COMMENTS:")
            for c, n in lost_src.items():
                print(f"    {c}")
        elif sum(sa.values()):
            print("  % src: comments: all survive")

    print("\nRESULT:", "VIOLATION" if bad else "no number changed, no src comment lost")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
