import torch
from torch.nn import Sequential

from org.campagnelab.dl.pytorch.cifar10.utils import init_params


class ModelAssembler:
    def __init__(self, num_features):
        self.num_features=num_features
        self.collect_output=None
        self.model=None

    def get_collect_output(self):
        return self.collect_output

    def assemble(self, activation_list):
        collect_output=[]
        reduced_output=[]
        index=0
        for activation in activation_list:
            included = True if activation.size()[1] <= 4096 else False
            collect_output.append(included)
            if included:

                reduced_output.append(activation)

                print("{} num activations: {}".format(index,
                                                      activation.size()))
            index += 1
        self.collect_output=collect_output

        flattened_activations = torch.cat(reduced_output, dim=1)
        num_activations=flattened_activations.size()[1]
        num_features=self.num_features

        self.model= Sequential(
            torch.nn.Linear(num_activations, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, 2),
            torch.nn.Softmax()
        )
        return self.model

    def evaluate(self, reduced_activation_list):
        #print("reduced_activation_list size: {} ".format(len(reduced_activation_list)))
        flattened_activations = torch.cat(reduced_activation_list, dim=1)
        return self.model(flattened_activations)

    def init_params(self):
        init_params(self.model)