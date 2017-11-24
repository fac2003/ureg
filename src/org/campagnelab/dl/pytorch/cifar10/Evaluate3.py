'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import random
import string
import sys

from torch.utils.data.sampler import RandomSampler

from org.campagnelab.dl.pytorch.cifar10.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.cifar10.Samplers import TrimSampler
from org.campagnelab.dl.pytorch.cifar10.TrainModel import TrainModel
from org.campagnelab.dl.pytorch.cifar10.models import *

# MIT License
#
# Copyright (c) 2017 Fabien Campagne
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

parser = argparse.ArgumentParser(description='Evaluate ureg against CIFAR10')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ureg', action='store_true', help='Enable unsupervised regularization (ureg)')
parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch', default=128)
parser.add_argument('--num-epochs', type=int,
                    help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
parser.add_argument('--num-training', '-n', type=int, help='Maximum number of training examples to use',
                    default=sys.maxsize)
parser.add_argument('--num-validation', '-x', type=int, help='Maximum number of training examples to use',
                    default=sys.maxsize)
parser.add_argument('--num-shaving', '-u', type=int, help='Maximum number of unlabeled examples to use when shaving'
                                                          'the network', default=sys.maxsize)
parser.add_argument('--max-examples-per-epoch', type=int, help='Maximum number of examples scanned in an epoch'
                                                                     '(e.g., for ureg model training).', default=None)

parser.add_argument('--ureg-num-features', type=int, help='Number of features in the ureg model', default=64)
parser.add_argument('--ureg-alpha', type=float, help='Mixing coefficient (between 0 and 1) for ureg loss component',
                    default=0.5)
parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                    default=''.join(random.choices(string.ascii_uppercase, k=5)))
parser.add_argument("--ureg-reset-every-n-epoch", type=int, help='Reset weights of the ureg model every n epochs.')
parser.add_argument('--ureg-learning-rate', default=0.01, type=float, help='ureg learning rate')
parser.add_argument('--shave-lr', default=0.1, type=float, help='shave learning rate')
parser.add_argument('--lr-patience', default=10, type=int,
                    help='number of epochs to wait before applying LR schedule when loss does not improve.')
parser.add_argument('--model', default="PreActResNet18", type=str,
                    help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')
parser.add_argument('--shaving-epochs', default=1, type=int,
                    help='number of shaving epochs.')
parser.add_argument('--drop-ureg-model', action='store_true',
                    help='Drop the ureg model at startup, only useful with --resume.')
parser.add_argument('--problem', default="CIFAR10", type=str,
                    help='The problem, either CIFAR10 or STL10')
parser.add_argument('--ureg-epsilon', default=1e-6, type=float, help='Epsilon to determine ureg model convergence.')

args = parser.parse_args()

print("Executing " + args.checkpoint_key)

use_cuda = torch.cuda.is_available()
is_parallel = False
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.problem=="CIFAR10":
    problem = Cifar10Problem(args.mini_batch_size)
elif args.problem=="STL10":
    problem = STL10Problem(args.mini_batch_size)
else:
    print("Unsupported problem: "+args.problem)
    exit(1)

# print some info about this dataset:
problem.describe()

model_trainer = TrainModel(args=args, problem=problem, use_cuda=use_cuda)

def vgg():
    return VGG('VGG16')


def resnet18():
    return ResNet18()


def preactresnet18():
    return PreActResNet18()


def googlenet():
    return GoogLeNet()


def densenet121():
    return DenseNet121()


def resnetx29():
    return ResNeXt29_2x64d()


def mobilenet():
    return MobileNet()


def dpn92():
    return DPN92()


def shufflenetg2():
    return ShuffleNetG2()


def senet18():
    return SENet18()


models = {
    "VGG16": vgg,
    "ResNet18": resnet18,
    "PreActResNet18": preactresnet18,
    "GoogLeNet": googlenet,
    "DenseNet121": densenet121,
    "ResNeXt29": resnetx29,
    "MobileNet": mobilenet,
    "DPN92": dpn92,
    "ShuffleNetG2": shufflenetg2,
    "SENet18": senet18
}


def create_model(name):
    function = models[args.model]
    if function is None:
        print("Wrong model name: " + args.model)
        exit(1)
    # construct the model specified on the command line:
    net = function()
    return net


model_trainer.init_model(create_model_function=create_model)

#model_trainer.training_combined()
model_trainer.training_interleaved(epsilon=args.ureg_epsilon)
