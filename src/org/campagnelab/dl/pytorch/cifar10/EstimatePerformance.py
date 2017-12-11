'''Estimate performance on the test set.'''
from __future__ import print_function

import argparse
import os

from org.campagnelab.dl.pytorch.cifar10.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.cifar10.TestModel import TestModel
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


def format_nice(n):
    try:
        if n == int(n):
            return str(n)
        if n == float(n):
            if n<0.001:
                return "{0:.3E}".format(n)
            else:
                return "{0:.4f}".format(n)
    except:
        return str(n)


parser = argparse.ArgumentParser(description='Estimate the performance of a model on the test set.')
parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
parser.add_argument('--fold-index', type=int, help='Index of the fold when testing a cross-validate model.',
                    default=None)

parser.add_argument('--checkpoint-key', help='random key to load a checkpoint model',
                    default=None)
parser.add_argument('--problem', default="CIFAR10", type=str,
                    help='The problem, either CIFAR10 or STL10')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()

print('==> Loading model from checkpoint..')
model = None
assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = None
try:
    if args.fold_index is None:
        checkpoint_filename='./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key)
    else:
        checkpoint_filename = './checkpoint/ckpt_{}-{}.t7'.format(args.checkpoint_key,
                                                                  args.fold_index)
    checkpoint = torch.load(checkpoint_filename)
except                FileNotFoundError:
    print("Unable to load model {} from checkpoint".format(args.checkpoint_key))
    exit(1)

if checkpoint is not None:
    model = checkpoint['net']
problem = None
if args.problem == "CIFAR10":
    problem = Cifar10Problem(args.mini_batch_size)
elif args.problem == "STL10":
    problem = STL10Problem(args.mini_batch_size)
else:
    print("Unsupported problem: " + args.problem)
    exit(1)

if problem is None or model is None:
    print("no problem or model, aborting")
    exit(1)

tester = TestModel(model=model, problem=problem, use_cuda=use_cuda, )
perfs = tester.test()
metrics = ["checkpoint"]

metric_values = [args.checkpoint_key]
for performance_estimator in perfs:
    metrics = metrics + performance_estimator.metric_names()
for performance_estimator in perfs:
    metric_values = metric_values + performance_estimator.estimates_of_metric()

with open("test-perfs-{}.tsv".format(args.checkpoint_key), "w") as perf_file:
    perf_file.write("\t".join(map(str, metrics)))
    perf_file.write("\n")
    perf_file.write("\t".join(map(format_nice, metric_values)))
    perf_file.write("\n")

print("\t".join(map(str, metrics)))
print("\t".join(map(format_nice, metric_values)))
exit(0)