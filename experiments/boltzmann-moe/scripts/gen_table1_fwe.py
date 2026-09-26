#!/usr/bin/env python3
"""Generate the FineWeb sweep table matching the user's Overleaf format.

Naming convention (from the user's reformatted table):
  - "Boltz-MoE rec"  = 6G1x6E recurrent Boltzmann
  - "Boltz-MoE deep" = 6S6E no-recurrence deep Boltzmann
  - "Switch MoE"     = 6G1x6S (dagger baseline)
  - "12-layer dense"  = 12G (dagger baseline)
  - "12N EGPT"       = 12N energy model, no MoE
  - "12S all-Switch"  = 12-layer all-Switch (dagger baseline)

Order: energy models first, baselines last (within each scale group).
Format: \\textit{energy} / \\textit{Baseline} sub-headers with \\midrule.
No MMLU column.  Best loss/Avg11/PPL bolded per scale group.
"""
import json, glob, os, re, sys

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fwe')

TASKS_NORM = ['arc_challenge', 'arc_easy', 'hellaswag', 'openbookqa', 'piqa', 'sciq']
TASKS_ACC  = ['boolq', 'copa', 'winogrande', 'race', 'lambada_openai']

GROUPS = [
    ('d=768, ~130M active, 4.0B tokens', [
        ('fineweb_E2_w1w2_sparse_K8',      'Boltz-MoE rec',        '6G1x6E', '8/2', False),
        ('fineweb_G6_6S6E_isoact_d768',    'Boltz-MoE deep',       '6S6E',   '4/2', False),
        ('fineweb_E3_switch_K8',            'Switch MoE',           '6G1x6S', '8/2', True),
        ('fineweb_A_12G_baseline',          '12-layer dense',       '12G',    '--',  True),
    ]),
    ('d=1024, ~225M active, 7.6B tokens', [
        ('fineweb_F5_12N_egpt_d1024',       '12N EGPT (no MoE)',    '12N',    '--',  False),
        ('fineweb_F2_6G1x6E_w1w2_sparse_K8_d1024', 'Boltz-MoE rec', '6G1x6E', '8/2', False),
        ('fineweb_G7_6S6E_isoact_d1024',    'Boltz-MoE deep',       '6S6E',   '4/2', False),
        ('fineweb_S1_12S_isoact_d1024',     '12S all-Switch',       '12S',    '4/2', True),
        ('fineweb_F3_6G1x6S_switch_K8_d1024', 'Switch MoE',        '6G1x6S', '8/2', True),
        ('fineweb_F1_12G_d1024',            '12-layer dense',       '12G',    '--',  True),
    ]),
    ('d=1280, ~315M active, 5.6B tokens', [
        ('fineweb_G81_6S6E_isoact_d1280_16gpu', 'Boltz-MoE deep',  '6S6E',   '4/2', False),
        ('fineweb_H2_6G1x6E_w1w2_sparse_K8_d1280', 'Boltz-MoE rec', '6G1x6E', '8/2', False),
        ('fineweb_H3_6G1x6S_switch_K8_d1280', 'Switch MoE',        '6G1x6S', '8/2', True),
        ('fineweb_H1_12G_d1280',            '12-layer dense',       '12G',    '--',  True),
    ]),
]

PARAMS = {
    'fineweb_E2_w1w2_sparse_K8':      ('163M', '132M', '194M'),
    'fineweb_G6_6S6E_isoact_d768':    ('166M', '132M', '132M'),
    'fineweb_E3_switch_K8':            ('163M', '132M', '196M'),
    'fineweb_A_12G_baseline':          ('162M', '162M', '162M'),
    'fineweb_F5_12N_egpt_d1024':       ('318M', '318M', '318M'),
    'fineweb_F2_6G1x6E_w1w2_sparse_K8_d1024': ('352M', '225M', '454M'),
    'fineweb_G7_6S6E_isoact_d1024':    ('309M', '225M', '225M'),
    'fineweb_S1_12S_isoact_d1024':     ('297M', '225M', '225M'),
    'fineweb_F3_6G1x6S_switch_K8_d1024': ('352M', '225M', '458M'),
    'fineweb_F1_12G_d1024':            ('255M', '255M', '255M'),
    'fineweb_G81_6S6E_isoact_d1280_16gpu': ('442M', '315M', '315M'),
    'fineweb_H2_6G1x6E_w1w2_sparse_K8_d1280': ('502M', '314M', '655M'),
    'fineweb_H3_6G1x6S_switch_K8_d1280': ('503M', '315M', '661M'),
    'fineweb_H1_12G_d1280':            ('363M', '363M', '363M'),
}


def find_eval(arm):
    d = os.path.join(RESULTS, arm)
    for pattern in ['**/harness_results_merged*.json', '**/harness_results*.json', '**/results_*.json']:
        jsons = sorted(glob.glob(os.path.join(d, pattern), recursive=True))
        if jsons:
            return jsons[-1]
    return None


def compute_avg11(jpath):
    with open(jpath) as f:
        data = json.load(f)
    vals = []
    for t in TASKS_NORM:
        v = data.get('results', {}).get(t, {}).get('acc_norm,none')
        if v: vals.append(v)
    for t in TASKS_ACC:
        v = data.get('results', {}).get(t, {}).get('acc,none')
        if v: vals.append(v)
    if len(vals) < 11:
        return None, None
    avg11 = sum(vals) / len(vals) * 100
    wppl = data.get('results', {}).get('wikitext', {}).get('word_perplexity,none')
    return avg11, wppl


def get_loss(arm):
    files = sorted(glob.glob(os.path.expanduser(f'~/bsub_logs/{arm}_*.stderr')), key=os.path.getmtime)
    if not files: return None
    last = None
    with open(files[-1]) as f:
        for line in f:
            m = re.search(r'train-lm_loss = ([0-9.]+)', line)
            if m: last = float(m.group(1))
    return last


def fmt(val, fmt_str, best_val=None):
    if val is None:
        return '--'
    s = format(val, fmt_str)
    if best_val is not None and val == best_val:
        return r'\textbf{' + s + '}'
    return s


def main_text():
    print("=" * 85)
    print("FineWeb Sweep Results (energy first, baselines last)")
    print("Avg11 = 11-task accuracy mean (pp, higher better). PPL = wikitext (lower).")
    print("=" * 85)

    for group_name, arms in GROUPS:
        print(f"\n### {group_name}")
        print(f"  {'Arm':<28} {'Arch':<7} {'K/k':<5} {'loss':>7} {'Avg11':>6} {'PPL':>5}")
        print("  " + "-" * 62)
        for arm, display, arch, kk, is_bl in arms:
            loss = get_loss(arm)
            jpath = find_eval(arm)
            avg11 = wppl = None
            if jpath:
                avg11, wppl = compute_avg11(jpath)
            bl = ' †' if is_bl else ''
            ls = f'{loss:.3f}' if loss else '--'
            a = f'{avg11:.2f}' if avg11 else '--'
            p = f'{wppl:.1f}' if wppl else '--'
            print(f"  {display + bl:<28} {arch:<7} {kk:<5} {ls:>7} {a:>6} {p:>5}")
    print("\n† = baseline")


def main_latex():
    for gi, (group_name, arms) in enumerate(GROUPS):
        losses, avg11s, ppls = [], [], []
        rows = []
        for arm, display, arch, kk, is_bl in arms:
            loss = get_loss(arm)
            jpath = find_eval(arm)
            avg11 = wppl = None
            if jpath:
                avg11, wppl = compute_avg11(jpath)
            total, active, flopwt = PARAMS.get(arm, ('--', '--', '--'))
            rows.append((arm, display, arch, kk, is_bl, total, active, flopwt, loss, avg11, wppl))
            if loss is not None: losses.append(loss)
            if avg11 is not None: avg11s.append(avg11)
            if wppl is not None: ppls.append(wppl)

        best_loss = min(losses) if losses else None
        best_avg = max(avg11s) if avg11s else None
        best_ppl = min(ppls) if ppls else None

        d_label = group_name.split(',')[0].strip()
        active_label = group_name.split(',')[1].strip()
        tok_label = group_name.split(',')[2].strip()

        print(r'\midrule')
        print(r'\multicolumn{10}{l}{{$' + d_label.replace('=', '{=}') +
              r'$, ${\sim}' + active_label.replace('~', '').replace('M active', r'$M active, $') +
              tok_label.replace('B tokens', r'$B tokens}} \\')
        print(r'\midrule')

        energy_rows = [r for r in rows if not r[4]]
        baseline_rows = [r for r in rows if r[4]]

        first_energy = True
        for arm, display, arch, kk, is_bl, total, active, flopwt, loss, avg11, wppl in energy_rows:
            prefix = r'\textit{energy} ' if first_energy else ''
            ls = fmt(loss, '.3f', best_loss)
            a = fmt(avg11, '.2f', best_avg)
            p = fmt(wppl, '.1f', best_ppl)
            print(f'{prefix} & {display} & {arch} & {kk} & {total} & {active} & {flopwt} & {ls} & {a} & {p} \\\\')
            first_energy = False

        if baseline_rows:
            print(r' \midrule')

        first_bl = True
        for arm, display, arch, kk, is_bl, total, active, flopwt, loss, avg11, wppl in baseline_rows:
            prefix = r'\textit{Baseline} ' if first_bl else ''
            bl = r' $\dagger$'
            ls = fmt(loss, '.3f', best_loss)
            a = fmt(avg11, '.2f', best_avg)
            p = fmt(wppl, '.1f', best_ppl)
            print(f'{prefix} & {display}{bl} & {arch} & {kk} & {total} & {active} & {flopwt} & {ls} & {a} & {p} \\\\')
            first_bl = False


def main():
    if '--latex' in sys.argv:
        main_latex()
    else:
        main_text()


if __name__ == '__main__':
    main()
