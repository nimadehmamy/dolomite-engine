# µP Scaling Probes for d=1024 Target

Sweep LR × width to find the µP-optimal LR that transfers across widths.

Protocol:
1. Run 500-step probes at d={256, 384, 512, 768} × LR={1e-2, 5e-3, 2e-3, 1e-3, 5e-4}
2. For each width, pick the LR with lowest final loss
3. Plot optimal_LR vs width — should be linear on log-log if µP holds
4. Extrapolate to d=1024

Architecture: 6G1x6E w1w2 sparse K=8 k=2 (matches E2)
Scale rules: I_mlp ∝ d, I_e ∝ d, heads = d/64, K fixed at 8
Data: nemotron-cc p2_0 (same as all FineWeb sweep runs)
