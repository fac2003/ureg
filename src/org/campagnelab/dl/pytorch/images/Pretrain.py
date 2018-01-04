'''Pretrain a model using the split image strategy.'''
from __future__ import print_function

import argparse
import copy
import random
import string
import sys

import numpy
from torchvision import models

from org.campagnelab.dl.pytorch.images.Cifar10NoTransformsProblem import Cifar10_NT64Problem
from org.campagnelab.dl.pytorch.images.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.images.CrossValidatedProblem import CrossValidatedProblem
from org.campagnelab.dl.pytorch.images.Problems import create_model
from org.campagnelab.dl.pytorch.images.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.images.STL10_NT64Problem import STL10_NT64Problem
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
    parser.add_argument('--pretrain', action='store_true', help='pretrain encoder and generator on unsupervised set.')

    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=128)
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--num-classes', type=int,
                        help='Number of classes to train with for pre-training..', default=10)
    parser.add_argument('--num-shaving', '-u', type=int,
                        help='Maximum number of unlabeled examples to use when shaving',
                        default=sys.maxsize)
    parser.add_argument('--num-validation', "-x", type=int, help='Maximum number of validation examples',
                        default=sys.maxsize)
    parser.add_argument('--num-training', '-n', type=int, help='Maximum number of training examples to use.',
                        default=sys.maxsize)

    parser.add_argument('--momentum', type=float, help='Momentum for SGD.', default=0.9)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--dropout', type=float, help='Dropout rate during pretraining, '
                                                      '0 for no dropout, 0.9 for 90% dropped.', default=0)

    parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--pretrained-key', help='key to load a pretrained model (use with --pretrain)',
                        default='DECONV')

    parser.add_argument('--lr-patience', default=10, type=int,
                        help='number of epochs to wait before applying LR schedule when loss does not improve.')
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')
    parser.add_argument('--mode', default="separate", type=str,
                        help='How to combine the unupervised reconstructed image and the training set image? separate keeps them '
                             'separate for training, average takes the average.')

    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem, either CIFAR10 or STL10')
    parser.add_argument('--num-cycles', type=int, help='Number of pre-training cycles.',
                        default=1000)
    parser.add_argument('--epochs-per-cycle', type=int, help='Number of epochs per cycle.',
                        default=10)
    parser.add_argument('--n-gpus', type=int, help='Use several gpus.', default=0)
    parser.add_argument('--max-accuracy', type=float, help='Maximum accuracy for early stopping a cycle.',
                        default=10.0)
    parser.add_argument('--num-encoder-features', type=int, help='Number of features used in the encoder.',
                        default=64)
    parser.add_argument('--num-generator-features', type=int, help='Number of features used in the generator.',
                        default=64)
    parser.add_argument('--num-representation-features', type=int,
                        help='Number of features used to represent an image.',
                        default=64)
    parser.add_argument('--constant-learning-rates', action='store_true',
                        help='Use constant learning rates, not schedules.')
    parser.add_argument('--cross-validation-folds', type=str,
                        help='Use cross-validation with folds defined in the argument file.'
                             ' The file follows the format of the STL-10 fold indices:'
                             ' one line per fold, with zero-based integers of the training examples in the train split.'
                             ' The validation examples are the complement indices in the train split.'
                             ' When this argument is provided, training is done sequentially for each fold, the '
                             ' checkpoint key is appended with the folder index and a summary performance file (cv-summary-[key].tsv) is written '
                             'at the completion of cross-validation. ', default=None)
    parser.add_argument('--cv-fold-min-perf', default=0, type=float,
                        help='Stop cross-validation early if a fold does not'
                             ' meet this minimum performance level (test accuracy).')
    parser.add_argument('--cross-validation-indices', type=str,
                        help='coma separated list of fold indices to evaluate. If the option '
                             'is not speficied, all folds are evaluated ', default=None)
    parser.add_argument('--seed', type=int,
                        help='Random seed', default=random.randint(0, sys.maxsize))

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("With CUDA")
    else:
        print("With CPU")

    is_parallel = False
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    if args.problem == "CIFAR10_NT64":
        problem = Cifar10_NT64Problem(args.mini_batch_size)
    elif args.problem == "STL10_NT64":
        problem = STL10_NT64Problem(args.mini_batch_size)
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

    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)

    if args.pretrain:
        print("Pre-training " + args.checkpoint_key)

        model_trainer = TrainModelDeconvolutionSplit(args=args, problem=problem, use_cuda=use_cuda)
        model_trainer.init_model(
            create_model_function=((lambda modelName, problem: nn.Linear(1, 1)) if args.pretrain else create_model))
        model_trainer.training_deconvolution()
        print("Finished pre-training " + args.checkpoint_key)
        exit(0)
    else:

        def get_metric_value(all_perfs, query_metric_name):
            for perf in all_perfs:
                metric = perf.get_metric(query_metric_name)
                if metric is not None:
                    return metric


        def train_once(args, problem, use_cuda):
            problem.describe()
            model_trainer = TrainModelDeconvolutionSplit(args=args, problem=problem, use_cuda=use_cuda)

            model_trainer.init_model(create_model_function=create_model)
            torch.manual_seed(args.seed)
            if use_cuda:
                torch.cuda.manual_seed(args.seed)

            model_trainer.init_model(create_model_function=create_model)

            return model_trainer.train_with_reconstructed_half()


        if args.cross_validation_folds is None:
            train_once(args, problem, use_cuda)
            exit(0)
        else:
            # load cross validation folds:
            fold_definitions = open(args.cross_validation_folds).readlines()
            initial_checkpoint_key = args.checkpoint_key
            all_perfs = []
            fold_indices = [int(index) for index in args.cross_validation_indices.split(",")] if \
                args.cross_validation_indices is not None else range(0, len(fold_definitions))

            for fold_index, fold in enumerate(fold_definitions):
                if fold_index in fold_indices:
                    splitted = fold.split(sep=" ")
                    splitted.remove("\n")
                    train_indices = [int(index) for index in splitted]
                    reduced_problem = CrossValidatedProblem(problem, train_indices)
                    args.checkpoint_key = initial_checkpoint_key + "-" + str(fold_index)
                    fold_perfs = train_once(copy.deepcopy(args), reduced_problem, use_cuda)

                    all_perfs += [copy.deepcopy(fold_perfs)]

                    if fold_perfs.get_metric("test_accuracy") < args.cv_fold_min_perf:
                        break

            metrics = ["train_loss", "test_loss", "test_accuracy"]
            accumulators = [0] * len(metrics)
            count = [0] * len(metrics)
            accuracies = []
            # aggregate statistics:
            for fold_perfs in all_perfs:
                for metric_index, metric_name in enumerate(metrics):
                    metric = get_metric_value(fold_perfs, metric_name)
                    if metric is not None:
                        print("found value for " + metric_name + " " + str(metric))
                        accumulators[metric_index] += metric
                        count[metric_index] += 1
                    if metric_name == "test_accuracy":
                        accuracies.append(metric)

            for metric_index, metric_name in enumerate(metrics):
                accumulators[metric_index] /= count[metric_index]
            test_accuracy_stdev = numpy.array(accuracies).std()
            with open("cv-summary-{}.tsv".format(initial_checkpoint_key), "w") as perf_file:
                perf_file.write("completed-folds\t")
                perf_file.write("\t".join(map(str, metrics)))
                perf_file.write("\ttest_accuracy_std")
                perf_file.write("\n")
                perf_file.write(str(len(all_perfs)))
                perf_file.write("\t")
                perf_file.write("\t".join(map(str, accumulators)))
                perf_file.write("\t")
                perf_file.write(str(test_accuracy_stdev))
                perf_file.write("\n")
