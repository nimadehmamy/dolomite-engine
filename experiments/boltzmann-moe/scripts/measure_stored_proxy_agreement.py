#!/usr/bin/env python
"""Measure the agreement of the proxy AS STORED IN A CHECKPOINT (HANDOFF §21.6).

`calibrate_proxy_router_20260916.py` FITS a fresh proxy and reports what a good fit achieves.
That is the wrong question when auditing an existing export: we need to know how good the proxy
that is actually in the file is, because that is the router the eval just ran.

Reported: top-k SET overlap between the STORED proxy's ranking and the EXACT all-K energies --
1.0 means the proxy picks exactly the experts the exact router would, k/K is chance.

  python scripts/measure_stored_proxy_agreement.py --run_dir <arm> --ckpt_name sparseeval_r16m512
"""
import sys, argparse, os
from pathlib import Path
import torch

sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "cal", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "calibrate_proxy_router_20260916.py"))
cal = importlib.util.module_from_spec(spec)
sys.modules["cal"] = cal
spec.loader.exec_module(cal)

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--ckpt_name", required=True)
ap.add_argument("--data_prefix",
                default="/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0")
ap.add_argument("--batches", type=int, default=4)
ap.add_argument("--seqlen", type=int, default=4096)
ap.add_argument("--per_call", type=int, default=1024)
a = ap.parse_args()

src = Path(a.run_dir) / a.ckpt_name
assert src.is_dir(), f"missing {src}"
from transformers import AutoModelForCausalLM
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
moes = [x for x in m.modules() if hasattr(x, "n_experts") and hasattr(x, "_fused_W")]
assert moes, "no fused mixture block here"
print(f"{src}")
print(f"  {len(moes)} block(s)  K={moes[0].n_experts}  top_k={moes[0].top_k}  "
      f"proxy_rank={getattr(moes[0],'proxy_rank',0)}  e_sign={moes[0].e_sign}  "
      f"sparse_active={getattr(moes[0],'_sparse_active',None)}")

cols = [cal.Collector(mo, a.per_call) for mo in moes]
seqs = cal.load_docs(a.data_prefix, a.batches, a.seqlen)
with torch.no_grad():
    for i in range(a.batches):
        m(input_ids=seqs[i:i + 1].to(dev))
data = [c.close() for c in cols]

print(f"\n  {'block':>5} {'tokens':>8} {'top-k agree':>12} {'chance':>8} {'verdict':>28}")
for bi, (mo, (x, E, it)) in enumerate(zip(moes, data)):
    k = int(mo.top_k) if mo.top_k else 1
    K = int(mo.n_experts)
    with torch.no_grad():
        # the STORED proxy, exactly as _forward_fused/_forward_sparse would call it
        Ep = mo._proxy_energies(x.to(next(mo.parameters()).dtype))
    ov = cal.agreement(Ep.double(), E.double(), k, mo.e_sign)
    chance = k / K
    if ov >= 0.85:      v = "trained, good"
    elif ov >= 0.5:     v = "trained, weak"
    elif ov <= chance * 1.6: v = "*** AT/NEAR CHANCE: UNFITTED ***"
    else:               v = "poor"
    print(f"  {bi:>5} {x.shape[0]:>8} {ov:>12.4f} {chance:>8.4f} {v:>28}")
