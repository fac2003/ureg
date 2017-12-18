'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from org.campagnelab.dl.pytorch.cifar10.models.feature_size_calculator import EstimateFeatureSize

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(EstimateFeatureSize):
    def __init__(self, vgg_name, input_shape=None):
        assert input_shape is not None, "You must provide the image input shape."
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.num_out = self.estimate_output_size_with_model(input_shape, self.features)
        self.remake_classifier(10, False)

    def remake_classifier(self, num_classes, use_cuda, dropout_p=0.5):
        self.classifier = nn.Sequential(nn.Linear(self.num_out, num_classes),
                                        nn.SELU(),
                                        nn.Linear(num_classes, num_classes),
                                        nn.SELU(),
                                        nn.Linear(num_classes, num_classes))
        if use_cuda: self.classifier = self.classifier.cuda()

    def get_classifier(self):
        return self.classifier

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
