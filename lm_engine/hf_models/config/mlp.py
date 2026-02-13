# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from typing import Any

from ...utils import BaseArgs


class _EnergyMLPArgs(BaseArgs):
    mlp_type: str = "Energy_MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "Energy_MLP"



class _MLPArgs(BaseArgs):
    mlp_type: str = "MLP"
    intermediate_size: int
    activation_function: str = "gelu_pytorch_tanh"
    dropout: float = 0
    add_bias: bool = False

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MLP"


class _MoEArgs(_MLPArgs):
    mlp_type: str = "MoE"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    use_interleaved_weights: bool = False
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE"


class _MoEEnergyArgs(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy"


class _MoEEnergyModuleArgs(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy_Module"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    energy_routing: bool = False        # True: route by expert energies (exact free-energy interpretation)
    energy_routing_tau: float = 1.0     # temperature for free-energy routing softmax

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy_Module"


class _MoEEnergyF5Args(_EnergyMLPArgs):
    mlp_type: str = "MoE_Energy_F5"
    shared_intermediate_size: int | None = None
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert_gating: bool = False
    normalized_topk: bool = True
    boltzmann_temperature: float = 1.0      # initial Boltzmann temperature
    learnable_temperature: bool = True       # make temperature a learnable parameter
    distillation_weight: float = 0.01        # weight for KL distillation loss
    use_boltzmann_at_inference: bool = False  # override to use Boltzmann routing at inference

    def model_post_init(self, __context: Any) -> None:
        assert self.mlp_type == "MoE_Energy_F5"
