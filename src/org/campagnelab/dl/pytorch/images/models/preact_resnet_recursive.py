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


def get_convolution(convolution_modules, in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False):
    try:
        return convolution_modules[(in_planes, planes, kernel_size, bias, stride, padding)]

    except:
        conv = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        convolution_modules[(in_planes, planes, kernel_size, bias, stride, padding)] = conv
        return conv


class PreActBlockRecursive(nn.Module):
    '''Recursive Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, convolution_modules, stride=1):
        super(PreActBlockRecursive, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = get_convolution(convolution_modules, in_planes, planes, kernel_size=3, stride=stride, padding=1,
                                     bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = get_convolution(convolution_modules, planes, planes, kernel_size=3, stride=1, padding=1,
                                     bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                get_convolution(convolution_modules, in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                                bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneckRecursive(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, convolution_modules, stride=1):
        super(PreActBottleneckRecursive, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = get_convolution(convolution_modules, in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = get_convolution(convolution_modules, planes, planes, kernel_size=3, stride=stride, padding=1,
                                     bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = get_convolution(convolution_modules, planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                get_convolution(convolution_modules, in_planes, self.expansion * planes, kernel_size=1, stride=stride,
                                bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNetRecursive(EstimateFeatureSize):
    def __init__(self, block, num_blocks, input_shape=(3, 32, 32), num_classes=10):
        super(PreActResNetRecursive, self).__init__()
        self.in_planes = 64
        self.convolution_modules = {}
        self.conv1 = get_convolution(convolution_modules=self.convolution_modules, in_planes=3, planes=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.blocks = {}
        self.layer1 = self._make_layer(block, 64, num_blocks[0],  stride=1,convolution_modules=self.convolution_modules)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,convolution_modules=self.convolution_modules)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,convolution_modules=self.convolution_modules)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,convolution_modules=self.convolution_modules)
        self.num_out = self.estimate_output_size(input_shape, self.features_forward)
        # self.features=nn.Sequential(self.conv1, self.layer1,self.layer2,self.layer3, self.layer4,F.avg_pool2d)
        self.remake_classifier(num_classes, False)

    def remake_classifier(self, num_classes, use_cuda, dropout_p=0.5):
        self.linear = nn.Sequential(nn.Linear(self.num_out, num_classes))
        if use_cuda: self.linear = self.linear.cuda()

    def get_classifier(self):
        return self.linear

    def _make_layer(self, block, planes, num_blocks, stride, convolution_modules):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes=planes, stride=stride,convolution_modules=convolution_modules))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def features_forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)

        return out


def PreActResNet18Recursive(input_shape):
    return PreActResNetRecursive(PreActBlockRecursive, [2, 2, 2, 2], input_shape)


def PreActResNet34Recursive(input_shape):
    return PreActResNetRecursive(PreActBlockRecursive, [3, 4, 6, 3], input_shape)


def PreActResNet50Recursive(input_shape):
    return PreActResNetRecursive(PreActBottleneckRecursive, [3, 4, 6, 3], input_shape)


def PreActResNet101Recursive(input_shape):
    return PreActResNetRecursive(PreActBottleneckRecursive, [3, 4, 23, 3], input_shape)


def PreActResNet152Recursive(input_shape):
    return PreActResNetRecursive(PreActBottleneckRecursive, [3, 8, 36, 3], input_shape)


def test():
    net = PreActResNet18Recursive(input_shape=(3, 32, 32))
    y = net(Variable(torch.randn(1, 3, 32, 32)))
    print(y.size())


#test()
