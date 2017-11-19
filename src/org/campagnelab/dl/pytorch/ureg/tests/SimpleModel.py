import torch
from torch.nn import Sequential


class SimpleModel(Sequential):
    def __init__(self, num_activations,num_features):
        super().__init__(
            torch.nn.Linear(num_activations, num_features),
            torch.nn.Sigmoid()
        )


