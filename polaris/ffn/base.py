"""Base class for feed-forward sub-layers."""

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class FeedForward(nn.Module, ABC):
    @abstractmethod
    def forward(self, x) -> tuple[Tensor, Tensor | None]:
        raise NotImplementedError
