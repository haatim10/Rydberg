#!/usr/bin/env bash
# PROMPT 14 A3 -- submission build gate.
#
# Compiles a manuscript and FAILS if anything unresolved reaches the PDF.
# This is the counterpart of the numerical gates and is not optional: a
# \todo{} that renders is a defect that ships.
#
# Usage:  scripts/submission_gate.sh paper/paper1/haatim_hsgs_letter
# Exit 0 = all gates pass. Exit 1 = at least one gate failed.
set -u

STEM="${1:?usage: submission_gate.sh <path/to/manuscript-without-extension>}"
DIR="$(dirname "$STEM")"
BASE="$(basename "$STEM")"
cd "$DIR" || exit 1

fail=0
say() { printf '  %-8s %s\n' "$1" "$2"; }

# ---------------------------------------------------------------- build
pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1
if [ -f "$BASE.bib" ] || [ -f refs.bib ]; then
  bibtex "$BASE" >/dev/null 2>&1
fi
pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1
pdflatex -interaction=nonstopmode "$BASE.tex" >/dev/null 2>&1

if [ ! -f "$BASE.pdf" ]; then
  say FAIL "no PDF produced"
  exit 1
fi

echo "=== submission gate: $BASE ==="

# ---- S1: nothing unresolved in the rendered text -----------------------
pdftotext "$BASE.pdf" - 2>/dev/null > /tmp/gate_text.txt
# Match the rendered forms only. '??' is how LaTeX prints a broken \ref.
hits=$(grep -oniE '\[TODO|\\todo|\bTODO\b|\?\?' /tmp/gate_text.txt | head -20)
if [ -n "$hits" ]; then
  say FAIL "S1 unresolved markers in the rendered PDF:"
  echo "$hits" | sed 's/^/           /'
  fail=1
else
  say PASS "S1 no TODO / \\todo / ?? in the rendered PDF"
fi

# ---- S2: page count ----------------------------------------------------
pages=$(pdfinfo "$BASE.pdf" 2>/dev/null | awk '/^Pages:/{print $2}')
if [ "${pages:-99}" -le 5 ]; then
  say PASS "S2 page count $pages <= 5"
else
  say FAIL "S2 page count $pages > 5"
  fail=1
fi

# ---- S4: every \cite resolves -----------------------------------------
und=$(grep -ci "undefined" "$BASE.log" 2>/dev/null || echo 0)
if [ "$und" -eq 0 ]; then
  say PASS "S4 no undefined references or citations"
else
  say FAIL "S4 $und undefined reference/citation warnings"
  grep -i "undefined" "$BASE.log" | head -5 | sed 's/^/           /'
  fail=1
fi

# ---- S6: no hardware claims -------------------------------------------
hw=$(grep -oniE 'measured hardware|recorded channel data|our (testbed|prototype)|vapou?r cell (measurement|experiment)' /tmp/gate_text.txt \
     | grep -viE 'no measured hardware|without measured hardware|no recorded channel data' | head -5)
if [ -z "$hw" ]; then
  say PASS "S6 no claim of measured hardware or recorded channel data"
else
  say FAIL "S6 possible hardware claim:"
  echo "$hw" | sed 's/^/           /'
  fail=1
fi

# ---- S7: the unstructured-LS reference is never an oracle or a bound ---
or=$(grep -oniE 'oracle|upper bound' /tmp/gate_text.txt | head -10)
if [ -z "$or" ]; then
  say PASS "S7 no 'oracle' or 'upper bound' anywhere"
else
  # Allowed only in explicitly negated form.
  bad=$(grep -niE 'oracle|upper bound' /tmp/gate_text.txt \
        | grep -viE 'no oracle|without an oracle|uses no oracle|not an upper bound|no .{0,12}upper bound')
  if [ -z "$bad" ]; then
    say PASS "S7 'oracle' appears only in negated form"
  else
    say FAIL "S7 unnegated 'oracle' / 'upper bound':"
    echo "$bad" | head -5 | sed 's/^/           /'
    fail=1
  fi
fi

# ---- S8: limitations paragraph intact ---------------------------------
miss=""
grep -qiE 'simulation only' /tmp/gate_text.txt || miss="$miss simulation-only"
grep -qiE 'no measured hardware' /tmp/gate_text.txt || miss="$miss no-hardware"
grep -qiE 'fundamental bound|no fundamental' /tmp/gate_text.txt || miss="$miss no-fundamental-bound"
if [ -z "$miss" ]; then
  say PASS "S8 limitations names simulation-only, no hardware, no fundamental bound"
else
  say FAIL "S8 limitations missing:$miss"
  fail=1
fi

# ---- S3 helper: numbers without a % src: comment -----------------------
# Advisory count only; S3 is verified per-claim in the report.
nsrc=$(grep -c '% src:' "$BASE.tex")
say INFO "S3 $nsrc '% src:' comments in the source (verified per-claim in the report)"

echo
if [ "$fail" -eq 0 ]; then
  echo "  ALL AUTOMATED GATES PASS"
else
  echo "  GATE FAILURE -- do not submit"
fi
exit "$fail"
