from dataclasses import dataclass
import torch
from torch import nn
from vpunfold.vpnet import VPLoss


@dataclass
class MethodParams:
    data: torch.Tensor
    rr: torch.Tensor
    initial_output: torch.Tensor
    target_output: torch.Tensor
    criterion: VPLoss
    device: torch.device
    model: nn.Module
    epsilon: float = 0.01


@dataclass
class SetupParams:
    model: nn.Module
    data: torch.Tensor
    rr: torch.Tensor
