# **************************************************
# Copyright (c) 2025
# Mixture of Energy_MLP Experts
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed._functional_collectives import all_reduce

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import ProcessGroupManager
from ...loss import add_aux_loss
from ...parameter import mark_parameter_as_mup_learning_rate
from ..dropout import Dropout
from ..linear import ParameterizedLinear
from .mlp import _get_std_for_linear, Energy_MLP
from .moe import ParameterizedExperts, compute_bincount


class MoE_Energy(nn.Module):
    """Mixture-of-Experts where each expert is an Energy_MLP.

    Energy_MLP forward per expert:
        W1x = W1(x)                                    # hidden -> intermediate
        W2x = W2(x)                                    # hidden -> intermediate
        y1 = gelu(W1x)
        y2 = sigmoid(scale * W1x) * 0.5 * W2x
        y2 = y2 @ W1.weight                            # intermediate -> hidden
        out = y1 @ W2.weight + y2                       # intermediate -> hidden

    Expert weights stored as two ParameterizedExperts:
        expert_W1: (num_experts, intermediate_size, hidden_size)
        expert_W2: (num_experts, intermediate_size, hidden_size)
    """

    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        normalized_topk: bool,
        shared_intermediate_size: int | None,
        shared_expert_gating: bool,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
    ) -> MoE_Energy:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.shared_expert_gating = shared_expert_gating
        self.normalized_topk = normalized_topk

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # Router gate: hidden_size -> num_experts
        self.gate = ParameterizedLinear(
            in_features=self.hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        # Optional shared expert gating (DeepSeek pattern)
        if self.shared_expert_gating:
            assert shared_intermediate_size is not None
            self.shared_expert_gate = ParameterizedLinear(
                in_features=self.hidden_size, out_features=1, bias=False, std=std
            )

        # Expert weight tensors for Energy_MLP dual-path forward
        # Each: (num_experts, intermediate_size, hidden_size)
        self.expert_W1 = ParameterizedExperts(
            num_experts=num_experts,
            in_features=hidden_size,
            out_features=intermediate_size,
            add_bias=add_bias,
            std=std,
        )
        self.expert_W2 = ParameterizedExperts(
            num_experts=num_experts,
            in_features=hidden_size,
            out_features=intermediate_size,
            add_bias=add_bias,
            std=std,
        )

        # Optional shared Energy_MLP expert (non-routed)
        if self.shared_intermediate_size is not None:
            self.shared_expert = Energy_MLP(
                hidden_size=hidden_size,
                intermediate_size=shared_intermediate_size,
                init_method=init_method,
                activation_function=activation_function,
                dropout=dropout,
                initializer_range=initializer_range,
                m_width=m_width,
                num_layers=num_layers,
                add_bias=add_bias,
            )

        self.dropout = Dropout(dropout)

        self.is_hopper_or_newer_gpu = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(torch.cuda.current_device()) >= (9, 0)
        )

        mark_parameter_as_mup_learning_rate(self.gate.weight)
        mark_parameter_as_mup_learning_rate(self.expert_W1.weight)
        mark_parameter_as_mup_learning_rate(self.expert_W2.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        moe_output, expert_frequency = self._compute_experts(hidden_states, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            hidden_states = moe_output
        else:
            hidden_states = moe_output + self._compute_shared_experts(hidden_states)

        del moe_output

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits,
                probs=torch.softmax(router_logits, dim=-1),
                expert_frequency=expert_frequency,
            )
            if self.training
            else 0
        )

        add_aux_loss(aux_loss)

        return hidden_states

    def _compute_routing_weights(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)

        if self.normalized_topk:
            router_weights, selected_experts = self._get_topk(router_logits)
            router_weights = F.softmax(router_weights.float(), dim=-1)
            router_weights = router_weights.type_as(hidden_states)
        else:
            router_weights = F.softmax(router_logits.float(), dim=-1)
            router_weights = router_weights.type_as(hidden_states)
            router_weights, selected_experts = self._get_topk(router_weights)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to experts, run Energy_MLP forward per expert, gather back."""

        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
            expert_frequency = compute_bincount(
                x=sorted_expert_idxs,
                size=self.num_experts,
                use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count),
            )

        T = hidden_states.size(0)
        batch_index = sorted_scattered_idxs // self.top_k

        # Gather gate values for sorted tokens
        batch_gates = router_weights.flatten()[sorted_scattered_idxs]

        # Gather input tokens in expert order
        x_sorted = hidden_states[batch_index]

        # Split by expert frequency
        freq_list = expert_frequency.tolist()
        x_splits = x_sorted.split(freq_list, dim=0)

        # Run Energy_MLP forward per expert
        outputs = []
        for i in range(self.num_experts):
            xi = x_splits[i]

            W1_i = self.expert_W1.weight[i]  # (intermediate_size, hidden_size)
            W2_i = self.expert_W2.weight[i]  # (intermediate_size, hidden_size)

            bias_W1 = None if self.expert_W1.bias is None else self.expert_W1.bias[i]
            bias_W2 = None if self.expert_W2.bias is None else self.expert_W2.bias[i]

            # Forward projections: hidden -> intermediate
            W1x = F.linear(xi, W1_i, bias_W1)
            W2x = F.linear(xi, W2_i, bias_W2)

            # GELU path
            y1 = F.gelu(W1x)

            # Sigmoid-gated path (energy gradient term)
            y2 = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5 * W2x
            y2 = y2 @ W1_i  # intermediate -> hidden (reuse W1)

            # Combine: y1 back-projected via W2 + y2
            out_i = y1 @ W2_i + y2  # intermediate -> hidden (reuse W2)

            outputs.append(out_i)

        hidden_states = torch.cat(outputs, dim=0)

        # Gate by routing weights
        hidden_states = hidden_states * batch_gates.unsqueeze(-1)

        # Scatter back to original token positions
        zeros = torch.zeros((T, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = zeros.index_add(0, batch_index, hidden_states)

        return hidden_states, expert_frequency

    def _compute_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = None
        if self.shared_expert_gating:
            gate = self.shared_expert_gate(hidden_states)

        output = self.shared_expert(hidden_states)

        if gate is not None:
            output = output * F.sigmoid(gate)

        return output

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)
        return x, indices

    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        expert_frequency = expert_frequency.float()

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            expert_frequency = all_reduce(
                expert_frequency, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group()
            )

        switch_loss = (
            num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()
        )
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss
        return loss.type_as(logits)

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Router-probability-weighted sum of per-expert energies."""
        x_flat = x.view(-1, self.hidden_size)
        router_logits = self.gate(x_flat)
        router_weights = F.softmax(router_logits.float(), dim=-1)
        router_weights = router_weights.type_as(x_flat)

        energy = torch.zeros(x_flat.size(0), device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            W1_i = self.expert_W1.weight[i]
            W1x = F.linear(x_flat, W1_i)
            expert_energy = F.gelu(W1x).sum(dim=-1)
            energy = energy + router_weights[:, i] * expert_energy

        if x.dim() == 3:
            energy = energy.view(x.shape[0], x.shape[1])
        return energy

    def get_num_active_parameters(self) -> int:
        num_elements = 0
        for parameter in self.parameters():
            num_elements += parameter.numel()

        for param_experts in [self.expert_W1, self.expert_W2]:
            for parameter in param_experts.parameters():
                num_elements -= parameter.numel()
                num_elements += (parameter.numel() * self.top_k) // self.num_experts

        return num_elements


class MoE_Energy_Module(nn.Module):
    """Mixture-of-Experts using actual Energy_MLP module instances as experts.

    Unlike MoE_Energy (which uses ParameterizedExperts 3D tensors), this variant
    stores each expert as a standalone Energy_MLP nn.Module — matching the standard
    MoE pattern where each expert is an independent module.

    Forward:
        1. Router: gate(x) -> top-k expert selection + routing weights
        2. Sort tokens by expert assignment
        3. For each expert, call expert.forward(tokens) directly
        4. Gate outputs by routing weights
        5. Scatter back to original token order
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        normalized_topk: bool,
        shared_intermediate_size: int | None,
        shared_expert_gating: bool,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        use_padding_free_transformer: bool,
    ) -> MoE_Energy_Module:
        super().__init__()

        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.use_padding_free_transformer = use_padding_free_transformer
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.shared_intermediate_size = shared_intermediate_size
        self.shared_expert_gating = shared_expert_gating
        self.normalized_topk = normalized_topk

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        expert_kwargs = dict(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            init_method=init_method,
            activation_function=activation_function,
            dropout=0,  # dropout applied after MoE aggregation, not per-expert
            initializer_range=initializer_range,
            m_width=m_width,
            num_layers=num_layers,
            add_bias=add_bias,
        )

        # Router gate
        self.gate = ParameterizedLinear(
            in_features=hidden_size,
            out_features=num_experts,
            bias=False,
            std=std,
        )

        # N independent Energy_MLP experts
        self.experts = nn.ModuleList([
            Energy_MLP(**expert_kwargs) for _ in range(num_experts)
        ])

        # Optional shared expert gating
        if self.shared_expert_gating:
            assert shared_intermediate_size is not None
            self.shared_expert_gate = ParameterizedLinear(
                in_features=hidden_size, out_features=1, bias=False, std=std
            )

        # Optional shared Energy_MLP (non-routed)
        if self.shared_intermediate_size is not None:
            self.shared_expert = Energy_MLP(
                **{**expert_kwargs, "intermediate_size": shared_intermediate_size}
            )

        self.dropout = Dropout(dropout)

        self.is_hopper_or_newer_gpu = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(torch.cuda.current_device()) >= (9, 0)
        )

        mark_parameter_as_mup_learning_rate(self.gate.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.use_padding_free_transformer:
            batch_size, sequence_length, _ = hidden_states.shape

        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = self._compute_routing_weights(hidden_states)
        moe_output, expert_frequency = self._compute_experts(hidden_states, router_weights, selected_experts)

        if self.shared_intermediate_size is None:
            hidden_states = moe_output
        else:
            hidden_states = moe_output + self._compute_shared_experts(hidden_states)

        del moe_output

        if not self.use_padding_free_transformer:
            hidden_states = hidden_states.reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = self.dropout(hidden_states)

        aux_loss = (
            self._compute_switch_loss(
                logits=router_logits,
                probs=torch.softmax(router_logits, dim=-1),
                expert_frequency=expert_frequency,
            )
            if self.training
            else 0
        )

        add_aux_loss(aux_loss)

        return hidden_states

    def _compute_routing_weights(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.gate(hidden_states)

        if self.normalized_topk:
            router_weights, selected_experts = self._get_topk(router_logits)
            router_weights = F.softmax(router_weights.float(), dim=-1)
            router_weights = router_weights.type_as(hidden_states)
        else:
            router_weights = F.softmax(router_logits.float(), dim=-1)
            router_weights = router_weights.type_as(hidden_states)
            router_weights, selected_experts = self._get_topk(router_weights)

        return router_logits, router_weights, selected_experts

    def _compute_experts(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch tokens to Energy_MLP module experts, gather back."""

        with torch.no_grad():
            sorted_expert_idxs, sorted_scattered_idxs = selected_experts.flatten().sort()
            expert_frequency = compute_bincount(
                x=sorted_expert_idxs,
                size=self.num_experts,
                use_continuous_count=self.is_hopper_or_newer_gpu and is_kernel_allowed(Kernel.continuous_count),
            )

        T = hidden_states.size(0)
        batch_index = sorted_scattered_idxs // self.top_k

        # Gather gate values for sorted tokens
        batch_gates = router_weights.flatten()[sorted_scattered_idxs]

        # Gather input tokens in expert order
        x_sorted = hidden_states[batch_index]

        # Split by expert frequency and route through each Energy_MLP module
        freq_list = expert_frequency.tolist()
        x_splits = x_sorted.split(freq_list, dim=0)

        outputs = []
        for i in range(self.num_experts):
            xi = x_splits[i]
            # Each expert is a full Energy_MLP module — just call forward
            outputs.append(self.experts[i](xi))

        hidden_states = torch.cat(outputs, dim=0)

        # Gate by routing weights
        hidden_states = hidden_states * batch_gates.unsqueeze(-1)

        # Scatter back to original token positions
        zeros = torch.zeros((T, self.hidden_size), dtype=hidden_states.dtype, device=hidden_states.device)
        hidden_states = zeros.index_add(0, batch_index, hidden_states)

        return hidden_states, expert_frequency

    def _compute_shared_experts(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate = None
        if self.shared_expert_gating:
            gate = self.shared_expert_gate(hidden_states)

        output = self.shared_expert(hidden_states)

        if gate is not None:
            output = output * F.sigmoid(gate)

        return output

    def _get_topk(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.top_k == 1:
            x, indices = x.max(dim=-1, keepdim=True)
        else:
            x, indices = x.topk(self.top_k, dim=-1)
        return x, indices

    def _compute_switch_loss(
        self, logits: torch.Tensor, probs: torch.Tensor, expert_frequency: torch.Tensor
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))

        num_experts = logits.size(1)
        acc_probs = probs.sum(0)

        expert_frequency = expert_frequency.float()

        if ProcessGroupManager.is_initialized() and ProcessGroupManager.get_data_parallel_world_size() > 1:
            expert_frequency = all_reduce(
                expert_frequency, reduceOp="sum", group=ProcessGroupManager.get_data_parallel_group()
            )

        switch_loss = (
            num_experts * (F.normalize(acc_probs, p=1, dim=0) * F.normalize(expert_frequency, p=1, dim=0)).sum()
        )
        z_loss = (torch.logsumexp(logits, dim=-1) ** 2).mean()

        loss = switch_loss + 0.1 * z_loss
        return loss.type_as(logits)

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Router-probability-weighted sum of per-expert energies."""
        x_flat = x.view(-1, self.hidden_size)
        router_logits = self.gate(x_flat)
        router_weights = F.softmax(router_logits.float(), dim=-1)
        router_weights = router_weights.type_as(x_flat)

        energy = torch.zeros(x_flat.size(0), device=x.device, dtype=x.dtype)
        for i in range(self.num_experts):
            expert_energy = self.experts[i].energy_per_token(x_flat.unsqueeze(0)).squeeze(0)
            energy = energy + router_weights[:, i] * expert_energy

        if x.dim() == 3:
            energy = energy.view(x.shape[0], x.shape[1])
        return energy

    def get_num_active_parameters(self) -> int:
        num_elements = 0
        for parameter in self.parameters():
            num_elements += parameter.numel()

        # Expert params: only top_k out of num_experts active per token
        expert_params = sum(p.numel() for expert in self.experts for p in expert.parameters())
        active_expert_params = (expert_params * self.top_k) // self.num_experts
        num_elements = num_elements - expert_params + active_expert_params

        return num_elements
