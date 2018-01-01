import torch
from torch.autograd import Variable
from torch.nn import Sequential

from org.campagnelab.dl.pytorch.images.utils import init_params


class ModelAssembler:
    def __init__(self, num_features, threshold_activation_size=None, layer_predicate_function=None):
        """ Construct a model assembler.
        :param num_features number of features in the assembled model.
        :param threshold_activation_size do not include layer outputs that have more activations than this threshold."""
        self.num_features = num_features
        self.collect_output = None
        self.model = None
        self.layer_predicate_function=lambda layer_index, activations: True
        
        if threshold_activation_size is not None and layer_predicate_function is None:
            self.threshold_activation_size=threshold_activation_size
            self.layer_predicate_function=lambda layer_index, activations: True \
                    if activations.size()[1] <= self.threshold_activation_size else False

        if layer_predicate_function is not None:
            self.layer_predicate_function=layer_predicate_function

    def get_collect_output(self):
        return self.collect_output

    def assemble(self, activation_list ):
        collect_output = []
        reduced_output = []
        index = 0

        for activation in activation_list:
            #included = True if activation.size()[1] <= self.threshold_activation_size else False
            included = self.layer_predicate_function(layer_index=index,activations=activation) if self.layer_predicate_function is not None else True
            collect_output.append(included)
            if included:
                reduced_output.append(activation)

                print("included layer {} with num activations: {}".format(index,
                                                      activation.size()))
            else:
                print("excluded layer {} with num activations {}".format(index,
                                                      activation.size()))
            index += 1
        assert len(collect_output)>0, "collected layer outputs must not be empty."
        self.collect_output = collect_output

        flattened_activations = torch.cat(reduced_output, dim=1)
        num_activations = flattened_activations.size()[1]
        num_features = self.num_features

        self.model = Sequential(
            torch.nn.Linear(num_activations, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, num_features),
            torch.nn.ReLU(),
            torch.nn.Linear(num_features, 2),
            torch.nn.Softmax()
        )
        return self.model

    def evaluate(self, reduced_activation_list):
        # print("reduced_activation_list size: {} ".format(len(reduced_activation_list)))
        flattened_activations = torch.cat(reduced_activation_list, dim=1)
        return self.model(flattened_activations)

    def evaluate_cat(self, reduced_activation_list_supervised, reduced_activation_list_unsupervised):
        # print("reduced_activation_list size: {} ".format(len(reduced_activation_list)))
        flattened_activations_sup = torch.cat(reduced_activation_list_supervised, dim=1)
        flattened_activations_unsup = torch.cat(reduced_activation_list_unsupervised, dim=1)
        flattened_activations_combined=Variable(torch.cat([flattened_activations_sup.data,flattened_activations_unsup.data]))
        return self.model(flattened_activations_combined)

    def init_params(self):
        init_params(self.model)
