'''Pre-activation ResNet in PyTorch.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

from org.campagnelab.dl.pytorch.images.models import EstimateFeatureSize
from org.campagnelab.dl.pytorch.images.models.dual import Dual, SequentialDual, LossEstimator_sim


class PreActBlockDual(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, loss_estimator=None):
        super(PreActBlockDual, self).__init__()
        assert loss_estimator is not None
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.is_dual = True
        self.conv1 = Dual(loss_estimator=loss_estimator,
                          delegate_module=nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                                    bias=False))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Dual(loss_estimator=loss_estimator,
                          delegate_module=nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Dual(loss_estimator=loss_estimator,
                                 delegate_module=nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                                                           stride=stride, bias=False))

    def forward(self, xs, xu):
        outs = F.relu(self.bn1(xs))
        outu = F.relu(self.bn1(xu)) if xu is not None else None
        (shortcuts, shortcutu, agreement_loss_shortcut) = self.shortcut(outs, outu) if hasattr(self,
                                                                                             'shortcut') \
            else (0.0, 0.0, 0.0)
        outs, outu, agreement_loss1 = self.conv1(outs, outu)
        outs, outu, agreement_loss2 = self.conv2(F.relu(self.bn2(outs)), F.relu(self.bn2(outu)) if outu is not None else None)

        outs += shortcuts
        if xu is not None: outu += shortcutu
        return outs, outu, agreement_loss1 + agreement_loss2 + agreement_loss_shortcut


class PreActBottleneckDual(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, loss_estimator=None):
        super(PreActBottleneckDual, self).__init__()
        assert loss_estimator is not None
        self.is_dual = True
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = Dual(loss_estimator=loss_estimator,
                          delegate_module=nn.Conv2d(in_planes, planes, kernel_size=1, bias=False))
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = Dual(loss_estimator=loss_estimator,
                          delegate_module=nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                                    bias=False))
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = Dual(loss_estimator=loss_estimator,
                          delegate_module=nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False))

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = SequentialDual(
                loss_estimator=loss_estimator,
                args=nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, xs, xu):
        outs = F.relu(self.bn1(xs))
        if xu is not None: outu = F.relu(self.bn1(xu))
        shortcuts, shortcutu, loss_shortcut = self.shortcut(outs, outu) if hasattr(self, 'shortcut') else 0
        outs, outu, loss_conv1 = self.conv1(outs, outu)
        outs, outu, loss_conv2 = self.conv2(F.relu(self.bn2(outs), F.relu(self.bn2(outu)) if outu is not None else None))
        outs, outu, loss_conv3 = self.conv3(F.relu(self.bn3(outs), F.relu(self.bn3(outu)) if outu is not None else None))
        outs += shortcuts
        outu += shortcutu
        return outs, outu, loss_shortcut + loss_conv1 + loss_conv2 + loss_conv3


class PreActResNetDual(EstimateFeatureSize):
    def __init__(self, block, num_blocks, input_shape=(3, 32, 32), num_classes=10, loss_estimator=None):
        super(PreActResNetDual, self).__init__()
        assert loss_estimator is not None
        self.in_planes = 64
        self.is_dual = True
        self.loss_estimator = loss_estimator
        self.conv1 = Dual(loss_estimator=loss_estimator,
                          delegate_module=nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False))
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, loss_estimator=loss_estimator)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, loss_estimator=loss_estimator)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, loss_estimator=loss_estimator)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, loss_estimator=loss_estimator)
        self.num_out = self.estimate_output_size_with_dual_function(input_shape, self.features_forward)
        # self.features=nn.Sequential(self.conv1, self.layer1,self.layer2,self.layer3, self.layer4,F.avg_pool2d)
        self.remake_classifier(num_classes, False)

    def remake_classifier(self, num_classes, use_cuda, dropout_p=0.5):
        self.linear = Dual(loss_estimator=self.loss_estimator, delegate_module=nn.Linear(self.num_out, num_classes))
        if use_cuda: self.linear = self.linear.cuda()

    def get_classifier(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride, loss_estimator):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, loss_estimator=loss_estimator))
            self.in_planes = planes * block.expansion
        return SequentialDual(loss_estimator=loss_estimator, args=layers)

    def forward(self, xs, xu):
        outs, outu, loss_conv1 = self.conv1(xs, xu)
        outs, outu, loss_layer1 = self.layer1(outs, outu)
        outs, outu, loss_layer2 = self.layer2(outs, outu)
        outs, outu, loss_layer3 = self.layer3(outs, outu)
        outs, outu, loss_layer4 = self.layer4(outs, outu)
        outs = F.avg_pool2d(outs, 4)
        if outu is not None: outu = F.avg_pool2d(outu, 4)
        outs = outs.view(outs.size(0), -1)
        if outu is not None: outu = outu.view(outu.size(0), -1)
        outs, outu, loss_linear = self.linear(outs, outu)
        return outs, outu, loss_conv1 + loss_layer1 + loss_layer2 + loss_layer3 + loss_layer4 + loss_linear

    def features_forward(self, xs, xu):
        outs, outu, loss_conv1 = self.conv1(xs, xu)
        outs, outu, loss_layer1 = self.layer1(outs, outu)
        outs, outu, loss_layer2 = self.layer2(outs, outu)
        outs, outu, loss_layer3 = self.layer3(outs, outu)
        outs, outu, loss_layer4 = self.layer4(outs, outu)
        outs = F.avg_pool2d(outs, 4)
        outu = F.avg_pool2d(outu, 4)
        outs = outs.view(outs.size(0), -1)
        outu = outu.view(outu.size(0), -1)

        return (outs, outu, 0)


def PreActResNet18Dual(input_shape, loss_estimator=LossEstimator_sim):
    return PreActResNetDual(PreActBlockDual, [2, 2, 2, 2], input_shape, loss_estimator=LossEstimator_sim)


def PreActResNet34Dual(input_shape):
    return PreActResNetDual(PreActBlockDual, [3, 4, 6, 3], input_shape, loss_estimator=LossEstimator_sim)


def PreActResNet50Dual(input_shape):
    return PreActResNetDual(PreActBottleneckDual, [3, 4, 6, 3], input_shape, loss_estimator=LossEstimator_sim)


def PreActResNet101Dual(input_shape):
    return PreActResNetDual(PreActBottleneckDual, [3, 4, 23, 3], input_shape, loss_estimator=LossEstimator_sim)


def PreActResNet152Dual(input_shape):
    return PreActResNetDual(PreActBottleneckDual, [3, 8, 36, 3], input_shape, loss_estimator=LossEstimator_sim)


def test():
    net = PreActResNet18Dual(input_shape=(3, 32, 32))
    xs = Variable(torch.randn(1, 3, 32, 32))
    xu = Variable(torch.randn(1, 3, 32, 32))
    ys,yu,loss = net(xs, xu)
    print(ys.size())


#test()
