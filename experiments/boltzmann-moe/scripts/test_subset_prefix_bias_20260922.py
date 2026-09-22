#!/usr/bin/env python
"""Is the rescued SUBSET distributionally different from the full corpus it prefixes?

WHY. The 2026-09-22 dataset rescue copied a byte-PREFIX of each web set (18.6% of
web-nemotron-cc-hq-p2_0/1) and a COMPLETE copy of each math set. A prefix is only an unbiased
sample if the corpus is in random order. If it is ordered -- by crawl, by source, by quality
bucket -- then every arm trained on the subset draws its web tokens from a biased slice and its
loss is NOT comparable to an arm trained on the full corpus.

This matters concretely: the 8B knob arms (subset) end at train-lm_loss ~2.33 while the 32B
cmix arms (full corpus) end at ~2.65, same model, same mixture weights, both schedules fully
decayed. Either the subset is easier or something else is going on, and the difference is far too
large (0.32 nats) to leave unexplained before running more arms on the copy.

METHOD, and why it needs no training. Take a checkpoint TRAINED ON THE FULL CORPUS and score it on
documents drawn from (a) inside the subset's span and (b) beyond it. The model saw both regions
during training, so any gap is a property of the DATA, not of exposure. Mean token NLL in nats,
lower = easier text.

  python scripts/test_subset_prefix_bias_20260922.py --ckpt <unsharded_dir>
"""
import argparse, sys
import torch

sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset

ORIG = "/proj/datasets/granite-4-datasets-megatron-merged"
SUB = "/proj/dmfexp/datasets-shared/granite-4-cmix-subset"
SETS = ["web-nemotron-cc-hq-p2_0", "web-nemotron-cc-hq-p2_1",
        "megamath-web-pro_0", "finemath-3plus-rewritten_0"]

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--seqs", type=int, default=24, help="sequences per region per dataset")
ap.add_argument("--seqlen", type=int, default=4096)
a = ap.parse_args()

from transformers import AutoModelForCausalLM
dev = "cuda" if torch.cuda.is_available() else "cpu"
m = AutoModelForCausalLM.from_pretrained(a.ckpt, trust_remote_code=True,
                                         torch_dtype=torch.bfloat16).to(dev).eval()
print(f"scoring model: {a.ckpt}\ndevice={dev}\n")


def grab(ds, lo, hi, n, seqlen, stride):
    """n sequences of seqlen tokens, from documents spread evenly over [lo, hi)."""
    out, buf, i = [], [], lo
    while len(out) < n and i < hi:
        try:
            buf.extend(ds[i].tolist())
        except Exception:
            pass
        i += stride
        while len(buf) >= seqlen and len(out) < n:
            out.append(buf[:seqlen]); buf = buf[seqlen:]
    return out


@torch.no_grad()
def nll(seqs):
    tot, ntok = 0.0, 0
    for s in seqs:
        ids = torch.tensor([s], device=dev)
        out = m(input_ids=ids, labels=ids)
        n = ids.shape[1] - 1
        tot += out.loss.float().item() * n; ntok += n
    return tot / max(ntok, 1)


print(f"  {'dataset':30s} {'span':>22s} {'docs':>12s} {'NLL (nats)':>11s}")
rows = []
for name in SETS:
    do = MMapIndexedDataset(f"{ORIG}/{name}")
    dsu = MMapIndexedDataset(f"{SUB}/{name}")
    N, n_sub = len(do), len(dsu)
    frac = 100.0 * n_sub / N
    # inside the subset's span, and beyond it; spread the sampling so we are not reading one
    # contiguous chunk of either region.
    ins = grab(do, 0, n_sub, a.seqs, a.seqlen, max(1, n_sub // (a.seqs * 4)))
    beyond = grab(do, n_sub, N, a.seqs, a.seqlen, max(1, (N - n_sub) // (a.seqs * 4))) if n_sub < N else []
    li = nll(ins) if ins else float("nan")
    print(f"  {name:30s} {'inside subset':>22s} {len(ins):>12d} {li:>11.4f}   (subset = {frac:.1f}% of docs)")
    if beyond:
        lb = nll(beyond)
        print(f"  {name:30s} {'BEYOND subset':>22s} {len(beyond):>12d} {lb:>11.4f}   delta = {li - lb:+.4f}")
        rows.append((name, li, lb))
    else:
        print(f"  {name:30s} {'(complete copy)':>22s} {'--':>12s} {'--':>11s}")

print("\n==================== VERDICT ====================")
if not rows:
    print("  every set is a complete copy -- no prefix bias possible")
else:
    for n, li, lb in rows:
        print(f"  {n:30s} inside {li:.4f}  beyond {lb:.4f}  delta {li-lb:+.4f} nats")
    mean = sum(li - lb for _, li, lb in rows) / len(rows)
    print(f"\n  mean delta (inside - beyond) = {mean:+.4f} nats over {len(rows)} web set(s)")
    print("  |delta| < 0.02  -> prefix is an unbiased sample; the subset is loss-comparable")
    print("  delta << 0      -> the subset's span is EASIER; subset losses are not comparable")
    print("                     to full-corpus losses and cross-copy comparisons are invalid")
