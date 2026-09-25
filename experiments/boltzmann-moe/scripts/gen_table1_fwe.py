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
"""
import json, glob, os, re, subprocess

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'fwe')

TASKS_NORM = ['arc_challenge', 'arc_easy', 'hellaswag', 'openbookqa', 'piqa', 'sciq']
TASKS_ACC  = ['boolq', 'copa', 'winogrande', 'race', 'lambada_openai']

# Arms organized by scale, energy first, baselines last
GROUPS = [
    ('d=768, ~130M active, 4.0B tokens', [
        ('fineweb_E2_w1w2_sparse_K8',      'Boltz-MoE rec',        '6G1x6E', '8/2', False),
        ('fineweb_G6_6S6E_isoact_d768',    'Boltz-MoE deep (iso)', '6S6E',   '4/2', False),
        ('fineweb_E3_switch_K8',            'Switch MoE',           '6G1x6S', '8/2', True),
        ('fineweb_A_12G_baseline',          '12-layer dense',       '12G',    '--',  True),
    ]),
    ('d=1024, ~225M active, 7.6B tokens', [
        ('fineweb_F5_12N_egpt_d1024',       '12N EGPT (no MoE)',    '12N',    '--',  False),
        ('fineweb_F2_6G1x6E_w1w2_sparse_K8_d1024', 'Boltz-MoE rec', '6G1x6E', '8/2', False),
        ('fineweb_G7_6S6E_isoact_d1024',    'Boltz-MoE deep (iso)', '6S6E',   '4/2', False),
        ('fineweb_S1_12S_isoact_d1024',     '12S all-Switch',       '12S',    '4/2', True),
        ('fineweb_F3_6G1x6S_switch_K8_d1024', 'Switch MoE',        '6G1x6S', '8/2', True),
        ('fineweb_F1_12G_d1024',            '12-layer dense',       '12G',    '--',  True),
    ]),
    ('d=1280, ~315M active, 5.6B tokens', [
        ('fineweb_G81_6S6E_isoact_d1280_16gpu', 'Boltz-MoE deep (iso)', '6S6E', '4/2', False),
        ('fineweb_H2_6G1x6E_w1w2_sparse_K8_d1280', 'Boltz-MoE rec', '6G1x6E', '8/2', False),
        ('fineweb_H3_6G1x6S_switch_K8_d1280', 'Switch MoE',        '6G1x6S', '8/2', True),
        ('fineweb_H1_12G_d1280',            '12-layer dense',       '12G',    '--',  True),
    ]),
]


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
        return None, None, None
    avg11 = sum(vals) / len(vals) * 100
    mmlu = data.get('results', {}).get('mmlu', {}).get('acc,none')
    mmlu = mmlu * 100 if mmlu else None
    wppl = data.get('results', {}).get('wikitext', {}).get('word_perplexity,none')
    return avg11, mmlu, wppl


def get_loss(arm):
    files = sorted(glob.glob(os.path.expanduser(f'~/bsub_logs/{arm}_*.stderr')), key=os.path.getmtime)
    if not files: return None
    with open(files[-1]) as f:
        for line in f:
            m = re.search(r'train-lm_loss = ([0-9.]+)', line)
            if m: last = float(m.group(1))
    return last if 'last' in dir() else None


def main():
    print("=" * 95)
    print("FineWeb Sweep Results (user format: energy first, baselines last)")
    print("Avg11 = 11-task accuracy mean (pp, higher better). MMLU separate. PPL = wikitext (lower).")
    print("=" * 95)

    for group_name, arms in GROUPS:
        print(f"\n### {group_name}")
        print(f"  {'Arm':<28} {'Arch':<7} {'K/k':<5} {'loss':>7} {'Avg11':>6} {'MMLU':>5} {'PPL':>5}")
        print("  " + "-" * 70)
        for arm, display, arch, kk, is_bl in arms:
            loss = get_loss(arm)
            jpath = find_eval(arm)
            if jpath:
                avg11, mmlu, wppl = compute_avg11(jpath)
            else:
                avg11 = mmlu = wppl = None
            bl = ' †' if is_bl else ''
            ls = f'{loss:.3f}' if loss else '--'
            a = f'{avg11:.2f}' if avg11 else '--'
            m = f'{mmlu:.1f}' if mmlu else '--'
            p = f'{wppl:.1f}' if wppl else '--'
            print(f"  {display + bl:<28} {arch:<7} {kk:<5} {ls:>7} {a:>6} {m:>5} {p:>5}")

    print("\n† = baseline")


if __name__ == '__main__':
    main()
