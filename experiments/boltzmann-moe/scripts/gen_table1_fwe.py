#!/usr/bin/env python3
"""Generate a results table for the FineWeb-Edu sweep experiments.

Adapted from gen_table1.py. Key differences:
  - Token budget is 4.0B (not 32.0B)
  - Data: nemotron-cc p2_0 (not cmix blend)
  - Config search: configs/iclr_26/fwe_sweep/
  - GROUPS organized by the experimental question, not by scale/recurrence
  - GSM8K skipped (not evaluated in these fast probes)
  - MMLU reported separately (same convention as the paper)

Run: python experiments/boltzmann-moe/scripts/gen_table1_fwe.py
  (CPU-only, no GPU needed — reads harness JSON files)
"""
import json, glob, os, sys

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results', 'fwe')

# Arms organized by question, not scale.
# (save_path_basename, display_name, is_baseline)
GROUPS = [
 ('Round 1: Legacy vs New code (d=768, 4.0B tokens)', [
   ('fineweb_A_12G_baseline',         '12G dense GPT baseline',         True),
   ('fineweb_B_legacy_w1w2_dense',    'Legacy w1w2 dense K=4',          False),
   ('fineweb_C_new_w1w2_dense',       'New w1w2 dense K=4',             False),
   ('fineweb_D_new_w1w2_sparse',      'New w1w2 sparse K=16 k=2',       False),
 ]),
 ('Round 2: Dense vs Sparse vs Switch, iso-total 162M (4.0B tokens)', [
   ('fineweb_E1_w1w2_dense_K8',       'w1w2 dense K=8',                 False),
   ('fineweb_E2_w1w2_sparse_K8',      'w1w2 sparse K=8 k=2',            False),
   ('fineweb_E3_switch_K8',           'Switch MoE K=8 k=2',              True),
   ('fineweb_D2_w1w2_sparse_isoflop', 'w1w2 sparse K=16 k=2 (iso-FLOP)', False),
 ]),
 ('Round 2: proj_type comparison (1.0B tokens)', [
   ('fineweb_E4_psdanti_probe',       'psd_anti',                        False),
   ('fineweb_E5_unconstrained_probe', 'unconstrained',                   False),
 ]),
]

# Where to find configs
SEARCH = ['configs/iclr_26/fwe_sweep']

def find_config(arm):
    """Find the YAML config for an arm."""
    for d in SEARCH:
        for f in glob.glob(os.path.join(d, '**', f'{arm}.yml'), recursive=True):
            return f
    return None

def find_eval(arm):
    """Find the eval results JSON for an arm."""
    d = os.path.join(RESULTS, arm)
    if not os.path.isdir(d):
        return None, 'no results dir'
    # Look for harness_results_*.json (not gsm8k_raw_*)
    jsons = sorted(glob.glob(os.path.join(d, '**/harness_results_*.json'), recursive=True))
    # Also check sparseeval and unsharded dirs
    for subdir in ['sparseeval_*', 'unsharded_*', 'eval_*', '']:
        pattern = os.path.join(d, subdir, 'harness_results_*.json') if subdir else os.path.join(d, 'harness_results_*.json')
        jsons.extend(glob.glob(pattern))
    if not jsons:
        return None, 'no harness_results JSON'
    return jsons[-1], None

def compute_avg11(results):
    """Compute Avg11 from a harness results dict."""
    TASKS_NORM = ['arc_challenge', 'arc_easy', 'hellaswag', 'openbookqa', 'piqa', 'sciq']
    TASKS_ACC  = ['boolq', 'copa', 'winogrande', 'race', 'lambada_openai']
    
    vals = []
    missing = []
    for t in TASKS_NORM:
        v = results.get('results', {}).get(t, {}).get('acc_norm,none')
        if v is not None:
            vals.append(v)
        else:
            missing.append(t)
    for t in TASKS_ACC:
        v = results.get('results', {}).get(t, {}).get('acc,none')
        if v is not None:
            vals.append(v)
        else:
            missing.append(t)
    
    if len(vals) < 11:
        return None, f'INCOMPLETE ({len(vals)}/11, missing: {missing})'
    return sum(vals) / len(vals) * 100, None  # percentage points

def get_mmlu(results):
    """Get MMLU accuracy."""
    v = results.get('results', {}).get('mmlu', {}).get('acc,none')
    if v is None:
        # Try the hendrycks format
        for k in results.get('results', {}):
            if 'mmlu' in k.lower() and 'acc' in str(results['results'][k]):
                v = results['results'][k].get('acc,none')
                if v is not None:
                    break
    return v * 100 if v is not None else None

def get_train_loss(arm):
    """Get final train-lm_loss from stderr."""
    import subprocess
    pattern = os.path.expanduser(f'~/bsub_logs/{arm}_*.stderr')
    files = sorted(glob.glob(pattern), key=os.path.getmtime)
    if not files:
        return None
    # Read last step line
    result = subprocess.run(['grep', 'train-lm_loss', files[-1]], capture_output=True, text=True)
    lines = result.stdout.strip().split('\n')
    if not lines or not lines[-1]:
        return None
    import re
    m = re.search(r'train-lm_loss = ([0-9.]+)', lines[-1])
    return float(m.group(1)) if m else None

def main():
    print("=" * 85)
    print("FineWeb-Edu Sweep Results")
    print("Data: nemotron-cc p2_0 (granite-4.0 tiktoken)")
    print("Avg11 = 11-task accuracy mean (pp, higher better)")
    print("MMLU = 5-shot accuracy (%, higher better, reported separately)")
    print("lm_loss = final training loss (nats, lower better)")
    print("=" * 85)
    print()
    
    for group_name, arms in GROUPS:
        print(f"### {group_name}")
        print(f"{'arm':<40} {'lm_loss':>8} {'Avg11':>8} {'MMLU':>8}")
        print("-" * 70)
        
        for arm, display, is_baseline in arms:
            loss = get_train_loss(arm)
            loss_str = f'{loss:.4f}' if loss else '—'
            
            jpath, err = find_eval(arm)
            if jpath:
                with open(jpath) as f:
                    data = json.load(f)
                avg11, e11 = compute_avg11(data)
                mmlu = get_mmlu(data)
                avg11_str = f'{avg11:.2f}' if avg11 else f'({e11})'
                mmlu_str = f'{mmlu:.2f}' if mmlu else '—'
            else:
                avg11_str = f'(no eval)'
                mmlu_str = '—'
            
            marker = ' *' if is_baseline else ''
            print(f"  {display:<38} {loss_str:>8} {avg11_str:>8} {mmlu_str:>8}{marker}")
        print()
    
    print("* = baseline")
    print()
    print("Note: Avg11 requires running eval_harness.py on the final checkpoint.")
    print("lm_loss is read from the training log and is available immediately.")

if __name__ == '__main__':
    main()
