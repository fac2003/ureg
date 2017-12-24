'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import collections
import random
import string

import sys
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.cifar10.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.cifar10.ConfusionTrainingHelper import ConfusionTrainingHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.Problems import create_model
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.cifar10.TrainModelSplit import TrainModelSplit
from org.campagnelab.dl.pytorch.cifar10.confusion.ConfusionModel import ConfusionModel
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
from org.campagnelab.dl.pytorch.cifar10.utils import batch




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use a confusion model to select unsupervised examples to supplement training.')
    parser.add_argument('--mini-batch-size', default=100, type=int, help='Size of the mini-batch.')
    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem that confusion data was collected for, either CIFAR10 or STL10')
    parser.add_argument('--checkpoint-key', help='random key to save/load the image model and save confusion models',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--max-examples', default=sys.maxsize, type=int, help='Scan at most max examples from the unuspervised set.')
    parser.add_argument('-n', default=1000, type=int, help='Return at most n examples per training_loss value.')

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    is_parallel = False
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    problem = None
    if args.problem == "CIFAR10":
        problem = Cifar10Problem(args.mini_batch_size)
    elif args.problem == "STL10":
        problem = STL10Problem(args.mini_batch_size)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    use_cuda=torch.cuda.is_available()
    print("Loading confusion model from {}".format(args.checkpoint_key))
    helper=ConfusionTrainingHelper(None, problem, args, use_cuda, checkpoint_key=args.checkpoint_key)

    priority_queues=helper.predict(max_examples=args.max_examples, max_queue_size=args.n)

    with open("unsupexamples-{}.tsv".format(args.checkpoint_key), mode="w") as unsup:
        for training_loss in helper.training_losses:
            unsup.write(str(training_loss))
            unsup.write("\t")
            unsup.write(" ".join(map(str,priority_queues.get(training_loss))))
            unsup.write("\n")

