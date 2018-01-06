'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
from torch.autograd import Variable

from org.campagnelab.dl.pytorch.images.models.dual import Dual
from org.campagnelab.dl.pytorch.images.models.feature_size_calculator import EstimateFeatureSize
from org.campagnelab.dl.pytorch.images.models.dual import SequentialDual

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGGDual(EstimateFeatureSize):

    """A dual version of the VGG models. """
    def __init__(self, vgg_name, input_shape=None,loss_estimator=None, use_cuda=False):
        super().__init__()
        assert input_shape is not None, "You must provide the image input shape."
        assert  loss_estimator is not None
        self.loss_estimator=loss_estimator

        self.features = self._make_layers(cfg[vgg_name],loss_estimator, use_cuda)
        self.num_out = self.estimate_output_size_with_dual_model(input_shape, model=self.features)
        self.remake_classifier(10, use_cuda)
        self.use_cuda = use_cuda



    def remake_classifier(self, num_classes, use_cuda, dropout_p=0.5):
        self.classifier = Dual(delegate_module=nn.Linear(self.num_out, num_classes), loss_estimator=self.loss_estimator)
        if use_cuda:
            self.classifier.cuda()

    def get_classifier(self):
        return self.classifier

    def forward(self, xs, xu):

        outs,outu,features_fm_loss = self.features(xs,xu)
        outs = outs.view(outs.size(0), -1)
        if outu is not None:
            outu = outu.view(outu.size(0), -1)
        outs, outu, classifier_fm_loss = self.classifier(outs, outu)
        return (outs, outu, features_fm_loss+classifier_fm_loss)

    def apply(self, fn):
        self.features.apply(fn)
        self.classifier.apply(fn)
    def _make_layers(self, cfg, loss_estimator, use_cuda):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=False)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return SequentialDual(loss_estimator=loss_estimator, args=layers)

# net = VGG('VGG11')
# x = torch.randn(2,3,32,32)
# print(net(Variable(x)).size())
