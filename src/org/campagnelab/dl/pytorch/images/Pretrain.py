'''Pretrain a model using the split image strategy.'''
from __future__ import print_function

import argparse
import copy
import random
import string
import sys

import numpy
from torchvision import models

from org.campagnelab.dl.pytorch.images.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.images.CrossValidatedProblem import CrossValidatedProblem
from org.campagnelab.dl.pytorch.images.Problems import create_model
from org.campagnelab.dl.pytorch.images.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.images.TrainModelDeconvolutionSplit import TrainModelDeconvolutionSplit
from org.campagnelab.dl.pytorch.images.TrainModelSplit import TrainModelSplit, flatten
from org.campagnelab.dl.pytorch.images.models import *

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate split training against CIFAR10 & STL10')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint.')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--num-classes', type=int,
                        help='Number of classes to train with for pre-training..', default=10)
    parser.add_argument('--num-shaving', '-u', type=int, help='Maximum number of unlabeled examples to use when shaving',
                        default = sys.maxsize)
    parser.add_argument('--num-validation', "-x", type=int, help='Maximum number of validation examples',
                                                               default=sys.maxsize)
    parser.add_argument('--momentum', type=float, help='Momentum for SGD.', default=0.9)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--dropout', type=float, help='Dropout rate during pretraining, '
                                                      '0 for no dropout, 0.9 for 90% dropped.', default=0)

    parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--lr-patience', default=10, type=int,
                        help='number of epochs to wait before applying LR schedule when loss does not improve.')
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')
    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem, either CIFAR10 or STL10')
    parser.add_argument('--num-cycles', type=int, help='Number of pre-training cycles.',
                        default=1000)
    parser.add_argument('--epochs-per-cycle', type=int, help='Number of epochs per cycle.',
                        default=10)
    parser.add_argument('--max-accuracy', type=float, help='Maximum accuracy for early stopping a cycle.',
                        default=10.0)
    parser.add_argument('--constant-learning-rates', action='store_true',
                        help='Use constant learning rates, not schedules.')

    args = parser.parse_args()

    print("Pre-training " + args.checkpoint_key)

    use_cuda = torch.cuda.is_available()
    if use_cuda: print("With CUDA")
    else: print("With CPU")

    is_parallel = False
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.problem == "CIFAR10":
        problem = Cifar10Problem(args.mini_batch_size)
    elif args.problem == "STL10":
        problem = STL10Problem(args.mini_batch_size)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)


    # print some info about this dataset:

    def get_metric_value(all_perfs, query_metric_name):
        for perf in all_perfs:
            metric = perf.get_metric(query_metric_name)
            if metric is not None:
                return metric

    problem.describe()

    model_trainer = TrainModelDeconvolutionSplit(args=args, problem=problem, use_cuda=use_cuda)
    if hasattr(args,'seed'):
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    model_trainer.init_model(create_model_function=create_model)
    model_trainer.training_deconvolution()

    print("Finished pre-training "+args.checkpoint_key)