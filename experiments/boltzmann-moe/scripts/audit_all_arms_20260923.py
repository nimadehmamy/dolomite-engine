import yaml
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config
ARMS=[
 ("134M hybrid 6G1x6E","configs/cmix/cmix_134M_hybrid_32B_sparse.yml"),
 ("134M hyb rnorm=none (abl_R)","configs/iclr_26/ablations/abl_R_134M_hyb_rnorm_none.yml"),
 ("134M block-pos 5G1x6E1G","configs/cmix/cmix_134M_sandwich_32B_sparse.yml"),
 ("134M blockpos+rnorm (abl_U)","configs/iclr_26/ablations/abl_U_134M_sandwich_rnorm_none.yml"),
 ("134M TRUE sandwich (abl_C)","configs/iclr_26/ablations/abl_C_134M_1G1x6E1G_isototal.yml"),
 ("134M w1w2 surrogate","configs/cmix/cmix_134M_hyb_w1w2_sparse_surr_32B.yml"),
 ("134M hopfield+surr (abl_S)","configs/iclr_26/ablations/abl_S_134M_hopfield_surrogate.yml"),
 ("134M pure rec 1x12E","configs/cmix/cmix_134M_pure_32B_sparse.yml"),
 ("134M abl_P K=32 (KILLED)","configs/iclr_26/ablations/abl_P_134M_pure_223_isoall_full.yml"),
 ("134M abl_P2 K=4 (NEW)","configs/iclr_26/ablations/abl_P2_134M_pure_223_K4.yml"),
 ("400M hybrid","configs/cmix/cmix_400M_hybrid_sparse.yml"),
 ("400M sandwich","configs/tok32B/t32B_sandwich_sparse.yml"),
 ("400M deep 6G6E (abl_H)","configs/iclr_26/ablations/abl_H_400M_6G6E_deep.yml"),
 ("1B stacked 8G4E","configs/cmix/cmix1B_12L_gptDense_32B.yml"),
]
print(f"{'arm':30s} {'hid':>5s} {'K':>4s} {'k':>3s} {'I_e':>7s} {'Ie/h':>5s} {'TOTAL':>8s} {'ACTIVE':>8s} {'FLOPwt':>8s}")
for tag,c in ARMS:
    try:
        pc=yaml.safe_load(open(c))['model_args']['pretrained_config']; h=pc['hidden_size']
        a=audit_config(c)
        ie=kk=K=None
        for b in pc['mlp_blocks']:
            if b.get('n_experts'):
                K=b['n_experts']; kk=b.get('top_k'); ie=b['intermediate_size']//K; break
        if ie is None:
            print(f"{tag:30s} {h:>5d} {'--':>4s} {'--':>3s} {'--':>7s} {'--':>5s} {a.total/1e6:>7.1f}M {a.active/1e6:>7.1f}M {a.flop_weight/1e6:>7.1f}M")
        else:
            print(f"{tag:30s} {h:>5d} {K:>4d} {kk:>3d} {ie:>7d} {ie/h:>5.2f} {a.total/1e6:>7.1f}M {a.active/1e6:>7.1f}M {a.flop_weight/1e6:>7.1f}M")
    except Exception as e:
        print(f"{tag:30s} ERROR {type(e).__name__}: {e}")
