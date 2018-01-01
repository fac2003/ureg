import torch
from torch.nn import Module, Sequential, Linear, ReLU

from org.campagnelab.dl.pytorch.images.utils import init_params


class ConfusionModel(Module):
    def __init__(self, image_model, problem):

        super().__init__()
        self.image_model=image_model
        self.num_classes=problem.num_classes()
        num_classes=self.num_classes
        num_inputs = image_model.num_out + 2
        self.classifier=Sequential(Linear(num_inputs, num_inputs),
                                   torch.nn.ReLU(True),
                                   torch.nn.Dropout(),
                                   Linear(num_inputs, num_inputs),
                                   torch.nn.ReLU(True),
                                   torch.nn.Dropout(),
                                   Linear(num_inputs, num_classes * num_classes))
        init_params(self.classifier)

    def forward(self, training_loss, trained_with, image_input):
        # combine training loss and image features before classifier:
        out=self.image_model.features_forward(image_input)
        out= torch.cat([out.view(image_input.size(0),-1),training_loss, trained_with],dim=1)
        out=self.classifier(out)
        return out