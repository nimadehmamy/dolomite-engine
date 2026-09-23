#!/bin/bash
# Regenerate BOTH generated tables in the ICLR draft, then verify the draft still compiles-clean.
#
# WHY A DRIVER (2026-09-22). The regeneration is four steps and skipping any one of them has
# already broken the paper once each: hand-splicing deleted a \resizebox line (unmatched brace,
# discoverable only on Overleaf, since pdflatex cannot run here); a splice deleted the caption and
# label because they sit AFTER the tabular; the caption kept quoting numbers the body no longer
# printed; and FLOPwt rounding made the caption's own iso-FLOP claim look false. Run this instead
# of doing it by hand.
#
# Does NOT commit and does NOT push -- it prints the diff for review.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
PAPER=$HOME/Code/overleaf/boltzmann-moe-ICLR-2026
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate 2>/dev/null
export PYTHONPATH=$REPO:${PYTHONPATH:-}
cd "$EXP"

echo "=== 1. arms that are complete but NOT yet in tab:main ==="
python3 scripts/gen_table1.py 2>/dev/null | grep -E "OMITTED" || echo "  (none omitted -- every listed arm is in)"

echo; echo "=== 2. splice tab:main into sec/experiments.tex ==="
python3 scripts/splice_generated_table.py "$PAPER/sec/experiments.tex" \
        "gen_table1.py --latex" -- python3 scripts/gen_table1.py --latex || exit 1

echo; echo "=== 3. splice tab:status into sec/appendix.tex ==="
python3 scripts/splice_generated_table.py "$PAPER/sec/appendix.tex" \
        "gen_status_table.py --latex" -- python3 scripts/gen_status_table.py --latex || exit 1

echo; echo "=== 4. caption numbers that are no longer in tab:main's body ==="
echo "    (derived quantities -- gaps, ratios, setup figures -- are expected here;"
echo "     a raw table value in this list means the caption went stale)"
python3 scripts/check_caption_numbers.py "$PAPER/sec/experiments.tex" tab:main 2>/dev/null \
  | sed -n '/NOT in body/,$p'

echo; echo "=== 4b. INDEPENDENT audit: re-derive every tab:main cell from source ==="
# Regenerating proves nothing on its own -- gen_table1.py and the table share code, so a bug
# there writes the same wrong number twice. audit_table1.py recomputes each cell by a separate
# path (audit_config for params, the 11 task scores re-averaged in place for Avg11, the arm own
# training log for lm_loss) and reads the .tex that will actually be submitted.
python3 scripts/audit_table1.py --tex "$PAPER/sec/experiments.tex" || \
  echo "  *** AUDIT FAILED -- DO NOT PUSH ***"

echo; echo "=== 5. whole-paper integrity ==="
cd "$PAPER"
python3 - <<'PY'
import re,glob
files=['main.tex']+sorted(glob.glob('sec/*.tex'))
labels=set()
for f in files: labels|=set(re.findall(r'\\label\{([^}]*)\}',open(f).read()))
keys=set()
for b in glob.glob('*.bib'): keys|=set(re.findall(r'@\w+\{([^,]+),',open(b,errors='ignore').read()))
bad=0
for f in files:
    raw=open(f).read()
    # STRIP COMMENTS FIRST. On 2026-09-22 this check reported BALANCED on a file whose table
    # was disabled, because it counted the `}` inside a commented-out `% }` line.
    # The lookbehind is (?<!\\) -- ONE backslash. With two it treats an ESCAPED \% as a
    # comment start, strips the rest of the line, and reports false mismatches.
    s='\n'.join(re.sub(r'(?<!\\)%.*$','',l) for l in raw.split('\n'))
    t=re.sub(r'\\[{}]','',s)
    ur=sorted(set(re.findall(r'\\(?:ref|eqref)\{([^}]*)\}',s))-labels)
    uc=set()
    for m in re.findall(r'\\cite[a-z]*\{([^}]*)\}',s): uc|={c.strip() for c in m.split(',')}
    uc=sorted(uc-keys)
    probs=[]
    if t.count('{')!=t.count('}'): probs.append(f"braces {t.count('{')}/{t.count('}')}")
    for e in ('table','tabular'):
        if len(re.findall(r'\\begin\{'+e+r'\*?\}',s))!=len(re.findall(r'\\end\{'+e+r'\*?\}',s)): probs.append(e)
    if ur: probs.append(f"undef-ref {ur}")
    if uc: probs.append(f"undef-cite {uc}")
    if probs: bad+=1; print(f"  *** {f}: {', '.join(probs)}")
w=[]
for f in files:
    s=open(f).read()
    for m in re.finditer(r'\\begin\{tabular\}\{((?:[^{}]|\{[^{}]*\})*)\}', s):
        n=len(re.findall(r'[lcrp]',re.sub(r'\{[^{}]*\}','',m.group(1))))
        if n>=7 and '\\resizebox' not in s[max(0,m.start()-700):m.start()]: w.append((f,n))
print(f"  files with problems: {bad};  wide tables missing resizebox: {w or 'none'}")
print("  ==> SAFE TO COMMIT" if bad==0 and not w else "  ==> DO NOT COMMIT")
PY

echo; echo "=== 6. diff ==="
git -C "$PAPER" diff --stat | sed 's/^/  /'
echo "  (review with: git -C $PAPER diff)"
