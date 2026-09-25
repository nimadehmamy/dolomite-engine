#!/usr/bin/env python3
"""Generate the MAIN RESULTS table (tab:main) -- the paper's headline table.

Usage: python scripts/gen_table1.py [--latex]

WHY A SCRIPT. The previous Table 1 (`tab:frontier`) was hand-maintained and went stale: it
reports the 7.86B-token wave on the OLD 100%-web Nemotron-CC mixture while the paper's claims
moved to 32.0B on the cmix 70/30 web/math blend. A hand table cannot be re-derived, so nobody
noticed. This regenerates from the configs and the stored evals, so it is either current or it
visibly omits an arm.

WHAT IT INCLUDES. Only MAIN architectures and their baselines -- hybrid, sandwich, pure, deep,
plus the Switch and dense references at each scale -- grouped 134M / 400M / 1B with a solid rule
between groups. Ablations stay in tab:status (the appendix status table).

THE 32B RULE. An arm appears ONLY if its checkpoint reached `num_training_steps` AND it has an
eval for that exact step. `metrics()` refuses results from any other step, because a stale
`unsharded_step4000` eval once reached the paper as a 1B arm's "32B" number (see
gen_status_table.py). An incomplete arm is OMITTED rather than shown with a partial number --
the caption asserts 32.0B for every row, so a row that is not 32B would falsify it.
"""
import yaml, json, glob, os, re, argparse, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = '/proj/dmfexp/nima/Code/dolomite-engine'
sys.path.insert(0, REPO)
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config

# scale -> [(config basename, display label, is_baseline)]
GROUPS = [
# RESTRUCTURED 2026-09-23: split by RECURRENCE, because FLOPwt/ACTIVE differs structurally between
# the two classes and mixing them made the 400M tier span FLOPwt 217-325 with no two rows
# iso-compute. Every group below is internally iso-ACTIVE and iso-FLOPwt (audit_config, <=1.1%):
#   134M recurrent      ACTIVE 123.2-123.5   FLOPwt 141.7-143.2
#   400M recurrent      ACTIVE 219.2-219.7   FLOPwt 299.1-301.0
#   400M no-recurrence  ACTIVE 238.0-239.5   FLOPwt 238.0-239.5
# A no-recurrence arm has FLOPwt == ACTIVE IDENTICALLY (every block applied once), so it can never
# sit at the recurrent group's 220/300. That is why the split is structural and not cosmetic, and
# why a no-rec row must NEVER be read against a recurrent row as iso-compute.
#
# DENSE baselines moved to Table 6 (user decision 2026-09-23: not essential here). All of them are
# already in gen_status_table.py ROLES, so nothing is lost: abl_D_134M_6G_dense_isototal,
# abl_F_134M_6G_dense_isoactive, abl_E_134M_6G1x6E_baseEGPT, abl_H_400M_6G6G_deep_isoactive.
# ALSO moved out: abl_U_134M_sandwich_rnorm_none (block-POSITION variant 5G1x6E1G, not a sandwich --
# reserved for the appendix), and cmix_400M_sandwich_sparse (layer_iterations [1,4,1] = 6 apps, so
# ACTIVE -29% and FLOPwt -27% against the hybrid; it is OFF this group's budget and is superseded by
# abl_AD_400M_sandwich_isoall).
 ('134M|Recurrent (ACTIVE 123.2M, FLOPwt 141.7M)', [
   ('cmix_134M_hybrid_32B_sparse',         'Hybrid (energy MoE)',            False),
   ('abl_R_134M_hyb_rnorm_none',           r'Hybrid, routing\_norm=none',    False),
   ('cmix_134M_sandwich_32B_sparse',       'Block-pos 5G1x6E1G (energy MoE)', False),
   ('abl_AA_134M_sandwich_isoall',         'Sandwich 1G1x6E1G, iso-all',     False),
   ('abl_B_134M_6G1x6S',                   'Switch MoE, FLOP-matched',       True),
   ]),
 ('400M|Recurrent (ACTIVE 219.4M, FLOPwt 299.4M)', [
   ('cmix_400M_hybrid_sparse',             'Hybrid (energy MoE)',            False),
   ('abl_X_400M_hyb_rnorm_none',           r'Hybrid, routing\_norm=none',    False),
   ('abl_AD_400M_sandwich_isoall',         'Sandwich 1G1x6E1G, iso-all',     False),
   ('abl_B_400M_6G1x6S',                   'Switch MoE, FLOP-matched',       True),
   ]),
 ('400M|No recurrence (ACTIVE 238.0M, FLOPwt 238.0M)', [
   ('abl_H_400M_6G6E_deep',                'Deep 6G6E (energy MoE)',         False),
   ('abl_AB_400M_6G6E_deep_rnormnone',     r'Deep 6G6E, routing\_norm=none', False),
   ('abl_AC_6S6E_deep_allmoe_isoactive',   'Deep 6S6E, all-MoE',             False),
   ('abl_H_400M_6G6S_deep',                'Switch MoE, deep',               True),
   ]),
 ('1B|No recurrence (ACTIVE 279.3M)', [
   ('cmix1B_12L_gptDense_32B',             'Stacked 8G4E (energy MoE)',      False),
   ]),
]


SEARCH = ['configs/cmix', 'configs/iclr_26/ablations', 'configs/iclr_26/scaling', 'configs/iclr_26']

def cfgpath(a):
    for d in SEARCH:
        p = os.path.join(REPO, d, a + '.yml')
        if os.path.exists(p): return p
    return None

def arch(pc):
    """Structural name: nG GPT block, nS Switch block, nE energy block; 1x6E = ONE block x6."""
    out=[]; blocks=pc.get('mlp_blocks') or []; iters=pc.get('layer_iterations') or []
    for i,b in enumerate(blocks):
        t=str(b.get('mlp_type',''))
        k='E' if 'EnergyFF_Boltzmann' in t else ('S' if ('MoE' in t or 'Switch' in t) else 'G')
        n=iters[i] if i < len(iters) else 1
        out.append((k,n))
    s=''; i=0
    while i < len(out):
        k,n = out[i]; j=i
        while j+1 < len(out) and out[j+1]==(k,n): j+=1
        cnt=j-i+1
        s += (f'{cnt}{k}' if n==1 else (f'{cnt}x{n}{k}' if cnt>1 else f'1x{n}{k}'))
        i=j+1
    return s

def metrics(sp, tot):
    """Eval for the FINAL step ONLY, from the DEPLOYED routing path.

    2026-09-23: this used to read `unsharded_*` exclusively, i.e. the ORACLE path -- all K expert
    energies computed and the cheap router bypassed. That is an upper bound the model cannot deliver
    at the FLOPs this table quotes, and it understated the method: abl_R printed 44.92 when its
    deployed number is 45.43 (exactly Switch parity). It also mixed provenance inside one column,
    because the dense and Switch rows have no cheap router and so were already showing their real
    numbers.

    Now: prefer `sparseeval_*` when the arm has one, else fall back to `unsharded_*`. A dense arm has
    no sparse/oracle distinction, so its single number IS its deployed number. The oracle figures
    stay in tab:status, which reports both paths side by side.

    EVERY metric in a row comes from the SAME export -- never mix a sparse Avg11 with an oracle
    GSM8K. A missing cell prints '--' rather than borrowing the other path's value.
    """
    cands = (f'{sp}/sparseeval_step{tot}', f'{sp}/sparseeval',
             f'{sp}/unsharded_step{tot}', f'{sp}/unsharded')
    for d in cands:
        c = glob.glob(f'{d}/harness_results_merged_*.json') or glob.glob(f'{d}/harness_results_2*.json')
        if c: break
    else:
        return None
    if not c: return None
    src = json.load(open(max(c, key=os.path.getmtime)))
    r = src.get('results', {})
    # gsm8k is a SEPARATE job (so a preemption cannot cost the 13-task pass), landing in <export>/gsm8k
    if 'gsm8k_cot' not in r:
        g = glob.glob(f'{d}/gsm8k/harness_results*.json')
        if g:
            try: r = {**r, **(json.load(open(max(g, key=os.path.getmtime))).get('results', {}))}
            except Exception: pass
    def m(task, *keys):
        d = r.get(task) or {}
        for k in keys:
            for kk, v in d.items():
                if kk == k or kk.startswith(k + ','): return v
        return None
    AN = ['arc_challenge','arc_easy','hellaswag','openbookqa','piqa','sciq']
    AC = ['boolq','copa','winogrande','race','lambada_openai']
    vals = [m(t,'acc_norm') for t in AN] + [m(t,'acc') for t in AC]
    if any(v is None for v in vals): return None
    return dict(avg11=100*sum(vals)/len(vals),
                ppl=m('wikitext','word_perplexity'),
                mmlu=(100*m('mmlu','acc')) if m('mmlu','acc') is not None else None,
                gsm=(100*(m('gsm8k_cot','exact_match,flexible-extract') or m('gsm8k_cot','exact_match') or 0))
                    if m('gsm8k_cot','exact_match') is not None or m('gsm8k_cot','exact_match,flexible-extract') is not None else None)


def tokens_per_step(arm, c):
    """Tokens/step for an arm, derived from ITS OWN LOG, never assumed.

    Order of preference: (1) `ddp=` in the DeviceMesh line x mbs x ga x seq, which is exact;
    (2) the empirical `billion_tokens_per_day * 1e9 * step_time / 86400`; (3) None.
    Returns None rather than guessing -- a wrong budget is worse than a missing one.
    """
    t = c['training_parameters']; seq = 4096
    mbs, ga = t['micro_batch_size'], t['gradient_accumulation_steps']
    # (0) THE LAUNCH LEDGER, added 2026-09-23. The log glob below keys on the arm's save_path
    # basename, but the LSF job name need not match it -- abl_U's save_path is
    # abl_U_134M_sandwich_rnorm_none while it was submitted as abl_U_sandwich_rnorm, so no log
    # matched and the row was dropped with 'cannot determine tokens/step'. submit_train.sh writes
    # gpus AND tokens_per_step into logs/launch_ledger.tsv at submission, keyed on the CONFIG path,
    # which is immune to the naming mismatch. Prefer the newest matching row.
    try:
        best = None
        for ln in open('logs/launch_ledger.tsv'):
            f = ln.rstrip('\n').split('\t')
            if len(f) < 16 or f[1] == 'jobid':
                continue
            if os.path.basename(f[3]).replace('.yml', '') and (arm in f[3] or arm in f[2]):
                try: best = int(f[14])
                except ValueError: pass
        if best:
            return best
    except OSError:
        pass
    logs = [f for f in glob.glob(os.path.expanduser(f'~/bsub_logs/{arm}_*.stderr'))
            if not os.path.basename(f).startswith(('ev_', 'evg_'))]
    for f in sorted(logs, key=os.path.getmtime, reverse=True):
        try: txt = open(f, errors='ignore').read()
        except Exception: continue
        m = re.search(r'DeviceMesh\(\(pp=\d+, ddp=(\d+)', txt)
        if m: return int(m.group(1)) * mbs * ga * seq
        e = re.findall(r'train-billion_tokens_per_day = ([0-9.]+).*?train-step_time \(sec\) = ([0-9.]+)', txt)
        if e:
            btd, st = float(e[-1][0]), float(e[-1][1])
            return int(round(btd * 1e9 * st / 86400 / 1024) * 1024)
    return None

def final_lm_loss(arm):
    """Last `train-lm_loss` in the arm's own training log, in nats.

    WHY THIS COLUMN EXISTS (2026-09-22). The Avg11 gaps in this table are much larger than the
    language-modelling loss can account for. At 134M the five non-pure arms sit inside 0.0338
    nats, which at the paper's ~4.52pp-per-nat slope predicts 0.15pp of Avg11 spread against
    2.10pp observed -- 13.7x. Without this column a reader takes a 0.73pp Avg11 lead as better
    language modelling, when the two arms differ by 0.024 nats. Note `train-lm_loss`, not
    `train-loss`: the latter includes aux losses and is not comparable across routing schemes.
    """
    logs = [f for f in glob.glob(os.path.expanduser(f'~/bsub_logs/{arm}_*.stderr'))
            if not os.path.basename(f).startswith(('ev_', 'evg_'))]
    # 2026-09-23: the glob keys on the arm's save_path basename, but the LSF job name need not
    # match it -- abl_U's save_path is abl_U_134M_sandwich_rnorm_none while it was submitted as
    # abl_U_sandwich_rnorm, so this returned None and the loss cell printed '--'. The launch ledger
    # records (name, jobid) against the CONFIG PATH, which reconstructs the real log filename.
    if not logs:
        try:
            for ln in open('logs/launch_ledger.tsv'):
                f2 = ln.rstrip('\n').split('\t')
                if len(f2) < 4 or f2[1] == 'jobid':
                    continue
                if arm in f2[3]:
                    logs += glob.glob(os.path.expanduser(f'~/bsub_logs/{f2[2]}_{f2[1]}.stderr'))
                    logs += glob.glob(os.path.expanduser(f'~/bsub_logs/{f2[2]}_*.stderr'))
        except OSError:
            pass
        logs = [f for f in dict.fromkeys(logs)
                if not os.path.basename(f).startswith(('ev_', 'evg_'))]
    for f in sorted(logs, key=os.path.getmtime, reverse=True):
        try: txt = open(f, errors='ignore').read()
        except Exception: continue
        v = re.findall(r'train-lm_loss = ([0-9.]+)', txt)
        if v: return float(v[-1])
    return None

def row(a, label, base):
    p = cfgpath(a)
    if not p: return None, f'{a}: config not found'
    c = yaml.safe_load(open(p)); pc = c['model_args']['pretrained_config']
    sp = c['save_args']['save_path']; tot = c['training_parameters']['num_training_steps']
    try: ck = json.load(open(sp+'/latest_checkpointed_iteration.json'))['latest_checkpointed_iteration']
    except Exception: ck = None
    if ck != tot: return None, f'{a}: INCOMPLETE {ck}/{tot} -- omitted (caption asserts 32.0B)'
    mm = metrics(sp, tot)
    if mm is None: return None, f'{a}: complete at {tot} but NO eval for that step -- omitted'
    r = audit_config(p)
    lml = final_lm_loss(a)
    # Energy blocks use n_experts/top_k; Switch (mlp_type: MoE) uses the HF names
    # num_experts/num_experts_per_tok. Checking only the former printed '--' for every Switch row.
    b = next((x for x in pc['mlp_blocks'] if 'n_experts' in x or 'num_experts' in x), None)
    if b is None:
        kk = '--'
    else:
        K = b.get('n_experts', b.get('num_experts'))
        k = b.get('top_k', b.get('num_experts_per_tok'))
        kk = f"{k}/{K}" if (K and k) else '--'
    # GPUS IS NOT IN THE CONFIG. Hardcoding 8 reported the 4-GPU abl_* baselines as 64.0B when
    # they are 32.0B -- the same trap that nearly ran abl_C at double budget. Read ddp= from the
    # arm's own log; fall back to the empirical billion_tokens_per_day x step_time.
    tps = tokens_per_step(a, c)
    if tps is None:
        return None, f'{a}: cannot determine tokens/step from its log -- omitted rather than guessed'
    tokB = tot*tps/1e9
    if abs(tokB - 32.0) > 0.6:
        return None, f'{a}: budget is {tokB:.1f}B, not 32.0B -- omitted (caption asserts 32.0B)'
    return dict(label=label, base=base, arch=arch(pc), kk=kk,
                act=r.active/1e6, tot_p=r.total/1e6, fw=r.flop_weight/1e6,
                tokens=tokB, lml=lml, **mm), None

def fmt(v, n=2):  return '--' if v is None else f'{v:.{n}f}'

ap = argparse.ArgumentParser(); ap.add_argument('--latex', action='store_true')
args = ap.parse_args()

data = []; notes = []
for scale, arms in GROUPS:
    rows = []
    for a, label, base in arms:
        r, why = row(a, label, base)
        if r: rows.append(r)
        else: notes.append(why)
    if rows: data.append((scale, rows))

if not args.latex:
    print(f"  {'scale':6s} {'arm':34s} {'arch':12s} {'k/K':>6s} {'Act/Tot':>14s} {'FLOPwt':>8s} {'Avg11':>6s} {'ppl':>7s} {'MMLU':>6s} {'GSM8K':>6s} {'lm_loss':>8s} {'tokens':>8s}")
    for scale, rows in data:
        for r in rows:
            _sz = scale.split("|")[0]
            print(f"  {_sz:5s} {r['label']:34s} {r['arch']:12s} {r['kk']:>6s} "
                  f"{r['act']:6.0f}M/{r['tot_p']:5.0f}M {r['fw']:7.1f}M {fmt(r['avg11']):>6s} "
                  f"{fmt(r['ppl']):>7s} {fmt(r['mmlu']):>6s} {fmt(r['gsm']):>6s} {fmt(r['lml'],4):>8s} {r['tokens']:7.1f}B")
    print()
    for n in notes: print(f"  OMITTED: {n}")
    sys.exit()

print(r"% GENERATED by experiments/boltzmann-moe/scripts/gen_table1.py --latex")
print(r"% Regenerate after any new 32B arm completes. Do NOT hand-edit.")
for n in notes: print(f"% omitted: {n}")
print(r"\resizebox{\textwidth}{!}{%")
print(r"\begin{tabular}{llcrrrrrrr}")
print(r"\toprule")
print(r"Architecture & Blocks & $k/K$ & Active/Total & FLOPwt & \texttt{lm\_loss} & Avg11 \% & WikiPPL & MMLU & GSM8K \\")
# Group labels are "SIZE|SUBGROUP". The SIZE header prints once per scale; each iso-compute
# subgroup (Recurrent / No recurrence) gets an indented sub-header under it. Bolding is
# WITHIN a subgroup, because only a subgroup is iso-ACTIVE and iso-FLOPwt -- a no-recurrence
# arm has FLOPwt == ACTIVE identically and can never share the recurrent budget.
_prev_size = None
for gi, (scale, rows) in enumerate(data):
    _size, _sub = (scale.split("|", 1) if "|" in scale else (scale, ""))
    if _size != _prev_size:
        print(r"\midrule")
        print(rf"\multicolumn{{10}}{{l}}{{\textbf{{{_size}}}}} \\")
        _prev_size = _size
    if _sub:
        print(rf"\multicolumn{{10}}{{l}}{{\quad\emph{{{_sub}}}}} \\")
    # A single-row group has no comparison to win, so nothing is bolded in it.
    cand = [r['avg11'] for r in rows if r['avg11'] is not None]
    best = max(cand) if len(cand) > 1 else None
    for r in rows:
        lab = r['label'] + (r' $\dagger$' if r['base'] else '')
        av = fmt(r['avg11'])
        if best is not None and r['avg11'] == best: av = rf"\mathbf{{{av}}}"; av = f"${av}$"
        print(rf"{lab} & \texttt{{{r['arch']}}} & {r['kk']} & "
              rf"{r['act']:.0f}M/{r['tot_p']:.0f}M & {r['fw']:.1f}M & {fmt(r['lml'],4)} & {av} & "
              rf"{fmt(r['ppl'])} & {fmt(r['mmlu'])} & {fmt(r['gsm'])} \\")
print(r"\bottomrule")
print(r"\end{tabular}%")
print(r"}")
