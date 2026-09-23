import copy, os, sys, tempfile, yaml
sys.path.insert(0,"/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config
base=yaml.safe_load(open("configs/iclr_26/ablations/abl_P2_134M_pure_223_K4.yml"))

def build(iters,I_tot,K,k,hidden=768,heads=12):
    c=copy.deepcopy(base); pc=c["model_args"]["pretrained_config"]
    pc["hidden_size"]=hidden; pc["num_layers"]=len(iters); pc["layer_iterations"]=list(iters)
    sm=copy.deepcopy(pc["sequence_mixer_blocks"][0])
    sm.update({"sequence_mixer_type":"energy_attention","num_attention_heads":heads,"num_key_value_heads":heads})
    pc["sequence_mixer_blocks"]=[copy.deepcopy(sm) for _ in iters]
    moe=[b for b in pc["mlp_blocks"] if b.get("n_experts")][0]
    out=[]
    for n in iters:
        b=copy.deepcopy(moe)
        b.update({"n_experts":K,"top_k":k,"intermediate_size":I_tot,
                  "sparse_candidates":k,"sinkhorn_mu_iters":int(n)})
        out.append(b)
    pc["mlp_blocks"]=out
    return c

def au(c):
    fd,p=tempfile.mkstemp(suffix=".yml"); os.close(fd)
    yaml.safe_dump(c,open(p,"w"),sort_keys=False)
    try:
        a=audit_config(p); return a.total/1e6,a.active/1e6,a.flop_weight/1e6
    finally: os.unlink(p)

CANDS=[("A  [2,2,3] h768 K8 k4",      (2,2,3),                      23288, 8, 4),
       ("A' [2,2,3] h768 K8 k5",      (2,2,3),                      23288, 8, 5),
       ("B  pure 10blk 14app K8 k2",  (1,1,1,1,1,1,2,2,2,2),        17920, 8, 2),
       ("B' pure 10blk 14app K16 k2", (1,1,1,1,1,1,2,2,2,2),        17920,16, 2),
       ("B'' pure 5blk 7app K8 k2",   (1,1,1,2,2),                  41984, 8, 2),
       ("B3 pure 10blk 14app K4 k2",  (1,1,1,1,1,1,2,2,2,2),        17920, 4, 2)]
print(f"  {'candidate':28s} {'apps/B':>6s} {'I_e':>6s} {'Ie/h':>5s} {'k/K':>5s} {'TOTAL':>8s} {'ACTIVE':>8s} {'FLOPwt':>8s}")
print(f"  {'TARGET (134M hybrid)':28s} {1.40:>6.2f} {1024:>6d} {1.33:>5.2f} {0.125:>5.3f} {134.25:>7.1f}M {123.24:>7.1f}M {141.72:>7.1f}M")
for nm,it,I,K,k in CANDS:
    ie=I//K
    t,a,f=au(build(it,I,K,k))
    print(f"  {nm:28s} {sum(it)/len(it):>6.2f} {ie:>6d} {ie/768:>5.2f} {k/K:>5.3f} {t:>7.1f}M {a:>7.1f}M {f:>7.1f}M"
          f"   dev T{100*(t/134.25-1):+.0f}% A{100*(a/123.24-1):+.0f}% F{100*(f/141.72-1):+.0f}%")
