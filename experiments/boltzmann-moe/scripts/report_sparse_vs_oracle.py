#!/usr/bin/env python
"""Oracle (dense all-K) vs proxy-sparse eval, per arm. See HANDOFF §21.

Every published number in this project was the ORACLE column: the checkpoint reloaded with
_sparse_active False and ran the dense all-K path. The SPARSE column is what the model can actually
deliver at its claimed FLOPs.

Prints a fully labelled table -- units and direction stated per column, per the project's table rule.

  python scripts/report_sparse_vs_oracle.py [--md]
"""
import json, glob, os, sys

ACC_NORM = ["arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa", "sciq"]
ACC = ["boolq", "copa", "winogrande", "race", "lambada_openai"]


def metrics(path):
    """Avg11 (pp), MMLU acc (pp), wikitext word-perplexity, GSM8K-CoT flexible-extract (pp)."""
    r = json.load(open(path))["results"]
    v = [(r.get(t) or {}).get("acc_norm,none") for t in ACC_NORM] + \
        [(r.get(t) or {}).get("acc,none") for t in ACC]
    avg11 = 100 * sum(v) / 11 if all(x is not None for x in v) else None
    mmlu = (r.get("mmlu") or {}).get("acc,none")
    wk = r.get("wikitext") or {}
    ppl = wk.get("word_perplexity,none")
    g = r.get("gsm8k_cot") or {}
    gsm = g.get("exact_match,flexible-extract")
    if gsm is None:
        gsm = g.get("exact_match,strict-match")
    return (avg11, 100 * mmlu if mmlu is not None else None, ppl,
            100 * gsm if gsm is not None else None)


def newest(pat):
    f = sorted(glob.glob(pat))
    return f[-1] if f else None


rows = []
for sd in sorted(glob.glob("results/**/sparseeval*", recursive=True)):
    if not os.path.isdir(sd):
        continue
    od = os.path.join(os.path.dirname(sd),
                      os.path.basename(sd).replace("sparseeval", "unsharded"))
    # The name-mangled guess is right for the gate-bug arms (sparseeval_step122070 <->
    # unsharded_step122070) but WRONG for the iclr_sink post-hoc exports, whose oracle sibling is
    # `unsharded` / `unsharded_mucal` / `unsharded_mucal2`, not `unsharded_r16m512`. The launch
    # ledger records the oracle dir explicitly, so prefer it and fall back to the guess.
    if not glob.glob(od + "/harness_results*.json"):
        try:
            for ln in open("logs/sparse_eval_ledger.tsv"):
                f = ln.rstrip("\n").split("\t")
                if len(f) >= 6 and f[4] and os.path.normpath(f[4]) == os.path.normpath(sd):
                    if glob.glob(f[5] + "/harness_results*.json"):
                        od = f[5]; break
        except OSError:
            pass
    sf, of = newest(sd + "/harness_results*.json"), newest(od + "/harness_results*.json")
    sg, og = newest(sd + "/gsm8k/harness_results*.json"), None
    for cand in (od + "/gsm8k_raw*.json", os.path.dirname(od) + "/gsm8k_step*/gsm8k_raw*.json"):
        og = og or newest(cand)
    if not (sf or of):
        continue
    s = metrics(sf) if sf else (None,) * 4
    o = metrics(of) if of else (None,) * 4
    # gsm8k is stored in a sibling dir; splice it in when the main file lacks it
    if s[3] is None and sg:
        s = s[:3] + (metrics(sg)[3],)
    if o[3] is None and og:
        try: o = o[:3] + (metrics(og)[3],)
        except Exception: pass
    arm = sd.replace("results/", "").replace("/sparseeval_step", " @").replace("/sparseeval", " @last")
    rows.append((arm, o, s))

md = "--md" in sys.argv
hdr = ["arm", "Avg11 orc", "Avg11 sps", "dAvg11", "MMLU orc", "MMLU sps", "ppl orc", "ppl sps", "GSM orc", "GSM sps"]
print("\nORACLE (dense all-K, what was published) vs PROXY-SPARSE (what the model delivers at its")
print("claimed FLOPs). Avg11 and MMLU are accuracy in PERCENTAGE POINTS, higher is better. ppl is")
print("wikitext WORD-level perplexity, LOWER is better. GSM is GSM8K-CoT flexible-extract exact")
print("match in pp, higher is better. dAvg11 = sparse - oracle in pp; NEGATIVE means sparsity costs")
print("accuracy. All rows are each arm's own final checkpoint on the cmix datamix.\n")
if md:
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))


def f(x, n=2):
    return "--" if x is None else f"{x:.{n}f}"


for arm, o, s in rows:
    d = (s[0] - o[0]) if (s[0] is not None and o[0] is not None) else None
    cells = [arm, f(o[0]), f(s[0]), (f"{d:+.2f}" if d is not None else "--"),
             f(o[1]), f(s[1]), f(o[2]), f(s[2]), f(o[3]), f(s[3])]
    print(("| " + " | ".join(cells) + " |") if md else
          f"{cells[0]:46s} " + " ".join(f"{c:>9s}" for c in cells[1:]))

done = [r for r in rows if r[2][0] is not None]
print(f"\n{len(done)}/{len(rows)} arm(s) have a completed sparse Avg11.")
if done:
    ds = [r[2][0] - r[1][0] for r in done if r[1][0] is not None]
    if ds:
        print(f"mean dAvg11 = {sum(ds)/len(ds):+.2f}pp   worst = {min(ds):+.2f}pp   best = {max(ds):+.2f}pp")
