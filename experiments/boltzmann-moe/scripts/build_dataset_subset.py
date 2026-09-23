#!/usr/bin/env python3
"""Build a smaller Megatron dataset by keeping a PREFIX of the documents of a big one.

WHY THIS IS EXACT AND NEEDS NO RE-TOKENISATION. In a Megatron `.bin` the sequences are stored
back-to-back in index order, and `_IndexWriter._sequence_pointers` recomputes the byte pointers
from the lengths starting at 0. Verified on the real files: `pointers[i] == sum(lengths[:i]) *
itemsize` for the first 100k sequences of both megamath and web p2_0. So keeping the first S
sequences means:
    new .bin = the first sum(lengths[:S]) * itemsize BYTES of the old .bin   (a plain byte copy)
    new .idx = _IndexWriter(lengths[:S], modes[:S], doc_indices <= S)
S is rounded UP to a document boundary so no document is cut in half.

WHAT THIS IS AND IS NOT FOR.
  IS:     running NEW 32B-token jobs on the same datamix, after the source is deleted.
  IS NOT: reproducing runs already published. `gpt_dataset.py` builds a `shuffle_index` that is a
          random permutation over the whole document set, so an existing run touched documents
          scattered across the full corpus. A subset changes which documents exist, hence which
          are sampled. Do not claim a rerun on this subset reproduces an earlier arm.

Usage:
  python scripts/build_dataset_subset.py --src <prefix> --dst <prefix> --tokens 20e9
  python scripts/build_dataset_subset.py --src ... --dst ... --tokens 1e9 --verify
"""
import argparse, os, subprocess, sys
import numpy as np
sys.path.insert(0, '/proj/dmfexp/nima/Code/dolomite-engine')
from lm_engine.data.megatron.indexed_dataset import _IndexReader, _IndexWriter, DType

ap = argparse.ArgumentParser()
ap.add_argument('--src', required=True, help='source prefix, without .bin/.idx')
ap.add_argument('--dst', required=True, help='destination prefix, without .bin/.idx')
ap.add_argument('--tokens', required=True, type=float, help='minimum tokens to keep, e.g. 20e9')
ap.add_argument('--verify', action='store_true', help='reload the result and compare documents')
a = ap.parse_args()
target = int(a.tokens)

r = _IndexReader(a.src + '.idx', multimodal=False)
# MEMORY: web p2_0 has 379,955,333 sequences. `sequence_lengths.astype(np.int64)` is 3.0 GB and a
# full `np.cumsum` another 3.0 GB, which killed this script at an 8 GB LSF limit. Keep the int32
# view from the mmap and accumulate in chunks instead; only the kept prefix is ever materialised.
L = r.sequence_lengths                      # int32 view onto the mmapped .idx, no copy
D = r.document_indices
itemsize = DType.size(r.dtype)
total = int(L.sum(dtype=np.int64))
print(f"  source {os.path.basename(a.src)}: {len(L):,} seqs, {len(D):,} doc bounds, "
      f"{total:,} tokens, dtype {np.dtype(r.dtype).name} ({itemsize} B/tok)")
if target >= total:
    sys.exit(f"  target {target:,} >= source {total:,}; copy the file whole instead of subsetting")

# find the first index whose cumulative token count reaches `target`, in 16M-element chunks
CH = 1 << 24
run = 0; need_seqs = None
for start in range(0, len(L), CH):
    blk = L[start:start + CH].astype(np.int64)
    c = np.cumsum(blk) + run
    hit = np.searchsorted(c, target, side='left')
    if hit < len(c):
        need_seqs = start + int(hit) + 1
        break
    run = int(c[-1])
    del blk, c
if need_seqs is None:
    sys.exit(f"  target {target:,} not reachable in {total:,} tokens")
# round UP to a document boundary so no document is truncated
pos = int(np.searchsorted(D, need_seqs, side='left'))
S = int(D[pos]) if pos < len(D) else int(D[-1])
if S < need_seqs:                                           # fell off the end
    S = int(D[-1])
kept_tokens = int(L[:S].sum(dtype=np.int64)) if S > 0 else 0
nbytes = kept_tokens * itemsize
# D is sorted ascending, so slice rather than mask: `D[D <= S]` would fault in all
# 379,955,334 int64 entries (3.0 GB) of the mmapped index.
Dnew = D[:int(np.searchsorted(D, S, side='right'))]
print(f"  keeping {S:,} seqs / {len(Dnew):,} doc bounds = {kept_tokens:,} tokens "
      f"({kept_tokens/target:.3f}x target) = {nbytes/2**30:.1f} GiB "
      f"[{100*kept_tokens/total:.2f}% of source]")

os.makedirs(os.path.dirname(a.dst) or '.', exist_ok=True)
# byte-exact prefix of the .bin; head -c is exact where dd with bs= may not be
print(f"  copying {nbytes:,} bytes of .bin ...", flush=True)
with open(a.dst + '.bin.part', 'wb') as out:
    p = subprocess.run(['head', '-c', str(nbytes), a.src + '.bin'], stdout=out)
if p.returncode != 0: sys.exit("  head failed")
got = os.path.getsize(a.dst + '.bin.part')
if got != nbytes: sys.exit(f"  .bin size wrong: got {got:,} want {nbytes:,}")
os.replace(a.dst + '.bin.part', a.dst + '.bin')

modes = getattr(r, 'sequence_modes', None)
with _IndexWriter(a.dst + '.idx', r.dtype) as w:
    # numpy arrays, NOT .tolist(): a 28M-element Python int list is ~1 GB of objects.
    # _IndexWriter tolerates arrays -- it uses len(), max(), np.array() and .item().
    w.write(L[:S].astype(np.int32), (modes[:S] if modes is not None else None), Dnew.astype(np.int64))
for ext in ('.ndocs',):
    if os.path.exists(a.src + ext):
        with open(a.dst + ext, 'w') as f: f.write(str(len(Dnew) - 1) + '\n')
print(f"  wrote {a.dst}.bin ({got/2**30:.1f} GiB) and {a.dst}.idx")

# --- self-check: the new index must describe the new .bin exactly
r2 = _IndexReader(a.dst + '.idx', multimodal=False)
L2 = r2.sequence_lengths.astype(np.int64)
assert len(L2) == S, f"seq count {len(L2)} != {S}"
assert int(L2.sum()) * itemsize == os.path.getsize(a.dst + '.bin'), "idx tokens != .bin bytes"
assert np.array_equal(L2, L[:S].astype(np.int64)), "lengths differ"
print(f"  self-check OK: {len(L2):,} seqs, {int(L2.sum()):,} tokens, idx and bin agree")

if a.verify:
    from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset
    src = MMapIndexedDataset(a.src, multimodal=False)
    dst = MMapIndexedDataset(a.dst, multimodal=False)
    print(f"  verify: len(src)={len(src):,} len(dst)={len(dst):,}")
    idxs = [0, 1, 2, S // 2, S - 2, S - 1]
    bad = 0
    for i in idxs:
        if i < 0 or i >= S: continue
        x, y = src[i], dst[i]
        same = len(x) == len(y) and bool(np.array_equal(np.asarray(x), np.asarray(y)))
        print(f"    doc {i:>12,}: len {len(y):>7} identical={same}")
        if not same: bad += 1
    print("  VERIFY PASS" if bad == 0 else f"  VERIFY FAILED on {bad} documents")
    sys.exit(1 if bad else 0)
