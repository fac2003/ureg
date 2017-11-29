'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import copy
import random
import string
import sys

from org.campagnelab.dl.pytorch.cifar10.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.cifar10.CrossValidatedProblem import CrossValidatedProblem
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem
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
parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint.')
parser.add_argument('--ureg', action='store_true', help='Enable unsupervised regularization (ureg).')
parser.add_argument('--constant-learning-rates', action='store_true', help='Use constant learning rates, not schedules.')

parser.add_argument('--constant-ureg-learning-rate', action='store_true', help='Keep ureg learning rate constant, do not use a schedule.')


parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
parser.add_argument('--num-epochs', '--max-epochs', type=int,
                    help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
parser.add_argument('--num-training', '-n', type=int, help='Maximum number of training examples to use.',
                    default=sys.maxsize)
parser.add_argument('--num-validation', '-x', type=int, help='Maximum number of training examples to use.',
                    default=sys.maxsize)
parser.add_argument('--num-shaving', '-u', type=int, help='Maximum number of unlabeled examples to use when shaving'
                                                          'the network.', default=sys.maxsize)
parser.add_argument('--max-examples-per-epoch', type=int, help='Maximum number of examples scanned in an epoch'
                                                               '(e.g., for ureg model training).', default=None)

parser.add_argument('--ureg-num-features', type=int, help='Number of features in the ureg model.', default=64)
parser.add_argument('--ureg-alpha', type=float, help='Mixing coefficient (between 0 and 1) for ureg loss component.',
                    default=0.5)
parser.add_argument('--momentum', type=float, help='Momentum for SGD.', default=0.9)
parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)

parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                    default=''.join(random.choices(string.ascii_uppercase, k=5)))
parser.add_argument("--ureg-reset-every-n-epoch", type=int, help='Reset weights of the ureg model every n epochs.')
parser.add_argument('--ureg-learning-rate', default=0.01, type=float, help='ureg learning rate.')
parser.add_argument('--shave-lr', default=0.1, type=float, help='shave learning rate.')
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
parser.add_argument('--mode', help='Training mode: combined or interleaved, used to alter the mode of '
                                   'ureg semi-supervised training.',
                    default="interleaved")
parser.add_argument('--threshold-activation-size', default=4096, type=int,
                    help='Do not include layers if they produce more than the specified number of activations.')
parser.add_argument('--grow-unsupervised-each-epoch', default=0, type=int,
                    help='Add the parameter to the number of unsupervised examples used after each epoch completes. '
                         'This parameter will grow --max-examples-per-epoch.')
parser.add_argument('--abort-when-failed-to-improve', default=sys.maxsize, type=int,
                    help='Abort training if performance fails to improve for more than the specified number of epochs.')

parser.add_argument('--test-every-n-epochs', type=int,
                    help='Estimate performance on the test set every n epochs. '
                         'Note that when test is skipped, the previous test '
                         'performances are reported in the log until new ones are available.', default=1)

parser.add_argument('--cross-validations-folds', type=str,
                    help='Use cross-validation with folds defined in the argument file.'
                         ' The file follows the format of the STL-10 fold indices:'
                         ' one line per fold, with zero-based integers of the training examples in the train split.'
                         ' The validation examples are the complement indices in the train split.'
                         ' When this argument is provided, training is done sequentially for each fold, the '
                         ' checkpoint key is appended with the folder index and a summary performance file (cv-summary-[key].tsv) is written '
                         'at the completion of cross-validation. ', default=None)
parser.add_argument('--cv-fold-min-perf', default=0, type=float, help='Stop cross-validation early if a fold does not'
                                                                      ' meet this minimum performance level (test accuracy).')

args = parser.parse_args()

print("Executing " + args.checkpoint_key)

with open("args-{}".format(args.checkpoint_key), "w") as args_file:
    args_file.write(" ".join(sys.argv))

use_cuda = torch.cuda.is_available()
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



def vgg16():
    return VGG('VGG16', problem.example_size())


def vgg19():
    return VGG('VGG19', problem.example_size())


def resnet18():
    return ResNet18(problem.example_size())


def preactresnet18():
    return PreActResNet18(problem.example_size())


def googlenet():
    return GoogLeNet()


def densenet121():
    return DenseNet121()


def resnetx29():
    return ResNeXt29_2x64d()


def mobilenet():
    return MobileNet()


def dpn92():
    return DPN92(problem.example_size())


def shufflenetg2():
    return ShuffleNetG2()


def senet18():
    return SENet18()


models = {
    "VGG16": vgg16,
    "VGG19": vgg19,
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


def get_metric_value(all_perfs, query_metric_name):
    for perf in all_perfs:
        metric = perf.get_metric(query_metric_name)
        if metric is not None:
            return metric


def train_once(args, problem, use_cuda):
    problem.describe()
    model_trainer = TrainModel(args=args, problem=problem, use_cuda=use_cuda)
    model_trainer.init_model(create_model_function=create_model)

    if args.mode == "combined":
        return model_trainer.training_combined()
    elif args.mode == "interleaved":
        return model_trainer.training_interleaved(epsilon=args.ureg_epsilon)
    else:
        print("unknown mode specified: " + args.mode)
        exit(1)


if args.cross_validations_folds is None:
    train_once(args, problem, use_cuda)
    exit(0)
else:
    # load cross validation folds:
    fold_definitions = open(args.cross_validations_folds).readlines()
    initial_checkpoint_key = args.checkpoint_key
    all_perfs = []
    for fold_index, fold in enumerate(fold_definitions):
        splitted = fold.split(sep=" ")
        splitted.remove("\n")
        train_indices = [int(index) for index in splitted]
        reduced_problem = CrossValidatedProblem(problem, train_indices)
        args.checkpoint_key = initial_checkpoint_key + "-" + str(fold_index)
        fold_perfs = train_once(copy.deepcopy(args), reduced_problem, use_cuda)

        all_perfs += [fold_perfs]

        if get_metric_value(fold_perfs,"test_accuracy") < args.cv_fold_min_perf:
            break



    metrics = ["train_loss", "train_accuracy", "test_loss", "test_accuracy"]
    accumulators = [0] * len(metrics)
    count=[0] * len(metrics)
    # aggregate statistics:
    for fold_perfs in all_perfs:
      for perf in fold_perfs:
        for metric_index, metric_name in enumerate(metrics):
            metric = perf.get_metric(metric_name)
            if metric is not None:
                # print("found value for "+metric_name+" "+str(metric))
                accumulators[metric_index] += metric
                count[metric_index] += 1

    for metric_index, metric_name in enumerate(metrics):
        accumulators[metric_index] /= count[metric_index]

    with open("cv-summary-{}.tsv".format(initial_checkpoint_key), "w") as perf_file:
        perf_file.write("completed-folds\t")
        perf_file.write("\t".join(map(str, metrics)))
        perf_file.write("\n")
        perf_file.write(str(len(all_perfs)))
        perf_file.write("\t")
        perf_file.write("\t".join(map(str, accumulators)))
        perf_file.write("\n")
