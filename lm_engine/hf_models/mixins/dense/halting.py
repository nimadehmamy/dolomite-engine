import torch
import torch.nn.functional as F
from torch import nn

from ...loss import add_aux_loss
from ...modeling_utils.activations import get_activation_function, is_glu
from ...modeling_utils.linear import ParameterizedLinear


# from ...parameter import mark_parameter_as_mup_learning_rate, mark_parameter_as_no_weight_decay


class HaltingGate(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        std: float,
        dropout: float = 0.05,
        aux_coeff: float = 1,
        ln: nn.Module = None,
        mlp_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        self.std = std
        self.dropout = dropout
        self.aux_coeff = aux_coeff
        if ln is None:
            ln = nn.RMSNorm(hidden_size)
        self.ln = ln
        self.is_mlp = mlp_intermediate_size is not None
        if self.is_mlp:
            self.in_transform = ParameterizedLinear(
                in_features=hidden_size, out_features=mlp_intermediate_size * 2, bias=False, std=self.std
            )
            self.act = get_activation_function("swiglu")
            # mark_parameter_as_mup_learning_rate(self.in_transform.weight)
        self.gate = ParameterizedLinear(
            in_features=hidden_size if not self.is_mlp else mlp_intermediate_size,
            out_features=2,
            bias=False,
            std=self.std,
        )
        # mark_parameter_as_mup_learning_rate(self.gate.weight)
        # self.register_buffer("noisy_bias", torch.tensor([0, 5.0]))
        self.reset_parameters()

    def extra_repr(self):
        return f"aux_coeff={self.aux_coeff},"

    def _gate_update(self, h, prev_log_omg):
        h = F.dropout(self.ln(h), p=self.dropout)
        if self.is_mlp:
            h = self.act(self.in_transform(h))
        z = self.gate(h).float() * 0.01

        if self.training:
            # random_halt = torch.rand_like(z[..., 0]) < self.dropout
            # selected_logits = torch.log_softmax(z[random_halt], dim=-1)
            # g = -torch.empty_like(selected_logits).exponential_().log()
            # z[random_halt] = (selected_logits + g) / 0.1
            log_p = torch.log_softmax(z, dim=-1)
            g = -torch.empty_like(log_p).exponential_().log()
            z = log_p + g
            z = z * 2.0

            # z[random_halt] += self.noisy_bias

            # count = int(h.size(1) * self.dropout)
            # random_halt = torch.rand_like(z[..., 0])
            # _, idxs = torch.topk(random_halt, k=count, dim=-1)
            # batch_idxs = torch.arange(h.size(0), dtype=torch.long, device=z.device)
            # selected_logits = z[batch_idxs[:, None], idxs]
            # # print(selected_logits)
            # # print(z[batch_idxs[:, None], idxs])
            # # z[random_halt] += self.noisy_bias
        else:
            z = z * 1e3
            # z = z * 2.0
            # z_flat = z.flatten(-2, -1)
            # halt_logits, halt_idx = torch.max(z_flat, dim=-1)
            # # z_flat = z_flat.index_add(dim=-1, index=halt_idx, source=5.)
            # idxs = torch.arange(z_flat.size(0), dtype=torch.long, device=z_flat.device)
            # z_flat[idxs[:, None], halt_idx] += 100.
            # z = z_flat.view(z.size())

        log_g_hat = torch.log_softmax(z, dim=-1)

        # print("halt > 0.5", (torch.exp(z[..., 0]) > 0.5).float().mean().item())
        if prev_log_omg is not None:
            cum_log_g = prev_log_omg[..., None] + log_g_hat
        else:
            cum_log_g = log_g_hat
        halted = torch.exp(cum_log_g[..., 0])
        return halted.to(h.dtype), cum_log_g[..., 1]

    def forward(self, prev_h, curr_h, prev_halt_state):
        if prev_halt_state is not None:
            prev_halted_h, prev_log_omg = prev_halt_state
        else:
            prev_halted_h, prev_log_omg = 0.0, None
        g, curr_log_omg = self._gate_update(prev_h, prev_log_omg)

        curr_halted_h = prev_halted_h + g[..., None] * prev_h
        # curr_aux_loss = prev_aux_loss + g
        # p = torch.clamp(torch.exp(curr_log_omg), max=1)  # prob of NOT halted =  1 - sum halted probabilities
        p = torch.exp(curr_log_omg)  # prob of NOT halted =  1 - sum halted probabilities
        # print(curr_halted_h[:, 0])
        # print(p)
        # print(curr_halted_h[:, 0] + p)
        # torch.set_printoptions(precision=0.2, linewidth=256)
        # print((p[0] > 0.4).long())
        # print(
        #     f"{(p < 0.25).float().mean().item():.5f} "
        #     f"{((0.25 < p) & (p < 0.50)).float().mean().item():.5f} "
        #     f"{((0.50 < p) & (p < 0.75)).float().mean().item():.5f} "
        #     f"{(0.75 < p).float().mean().item():.5f} "
        #     f"| Mean p: {p.mean().item():.5f}"
        # )

        # FIXME halting loss
        if self.training:
            add_aux_loss(self.aux_coeff * p.mean())
            curr_h = torch.where((p < 0.01)[..., None], prev_h, curr_h)

        unhalted = p[..., None].to(curr_h.dtype) * curr_h
        s = curr_halted_h + unhalted
        return s, p, (curr_halted_h, curr_log_omg)

    def finalise(self, curr_h, prev_halt_state):
        if prev_halt_state is not None:
            prev_halted_h, prev_log_omg = prev_halt_state
        g, _ = self._gate_update(curr_h, prev_log_omg)
        curr_halted_h = prev_halted_h + g[..., None] * curr_h
        return curr_halted_h

    def reset_parameters(self):
        nn.init.zeros_(self.gate.weight)
        if self.gate.bias is not None:
            nn.init.zeros_(self.gate.bias)

    # def compute_loss(self, halt_state):
    #     _, log_omg, aux_loss = halt_state