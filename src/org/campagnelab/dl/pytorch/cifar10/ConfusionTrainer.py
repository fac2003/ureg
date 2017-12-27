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
    parser = argparse.ArgumentParser(description='Train a confusion model.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--mini-batch-size', default=100, type=int, help='Size of the mini-batch.')
    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem that confusion data was collected for, either CIFAR10 or STL10')
    parser.add_argument('--confusion-data', type=str,
                        help='File with confusion data to train the model with.')
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to train before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--max-training', type=int,
                        help='Number of training example to use at each epoch.', default=sys.maxsize)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')
    parser.add_argument('--checkpoint-key', help='random key to save/load the image model and save confusion models',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--progress-bar', action='store_true', help='Show progress bars')
    parser.add_argument('--score-threshold', type=float, help='Train only with validation scores below the threshold.')
    args = parser.parse_args()

    Confusion = collections.namedtuple('Confusion',
                                       'trained_with example_index epoch train_loss predicted_label true_label val_loss')
    print("Loading confusion data..")
    confusion_data = []
    best_score=sys.maxsize
    with open(args.confusion_data, mode="r") as conf_data:
        for line in conf_data.readlines():
            line = line.replace("\n", "")
            trained_with, example_index, epoch, train_loss, predicted_label, true_label, val_loss = line.split("\t")
            best_score=min(best_score,float(val_loss))
            true_label = true_label.split("\n")[0]
            confusion_data += [Confusion(bool(trained_with == "True"), int(example_index), int(epoch), \
                                             float(train_loss), int(predicted_label), int(true_label), float(val_loss))]
    distinct_validation_losses = list(set([cd.val_loss for cd in confusion_data]))
    distinct_validation_losses.sort()
    distinct_validation_losses.reverse()
    print("Read the following validation losses: " + " ".join(map(str, distinct_validation_losses)))

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

    if args.score_threshold is not None:
        confusion_data=[c for c in confusion_data if c.val_loss<=args.score_threshold]

    use_cuda = torch.cuda.is_available()
    print("Loaded {} lines of confusion data".format(len(confusion_data)))

    print("Loading pre-trained image model from {}".format(args.checkpoint_key))
    image_model = TrainModelSplit(args, problem, use_cuda).load_checkpoint()

    helper = ConfusionTrainingHelper(image_model, problem, args, use_cuda)
    random.shuffle(confusion_data)
    train_split = confusion_data[0:int(len(confusion_data) * 2 / 3)]
    test_split = confusion_data[int(len(confusion_data) / 3):len(confusion_data)]
    best_loss = sys.maxsize
    no_improvement = 0

    distinct_training_losses = set([cd.train_loss for cd in confusion_data])
    distinct_validation_losses = set([cd.val_loss for cd in confusion_data])
    for epoch in range(0, args.num_epochs):
        perfs = PerformanceList()
        perfs += [helper.train(epoch, train_split)]
        perfs += [helper.test(epoch, test_split)]

        train_loss = perfs.get_metric("train_loss")
        test_loss = perfs.get_metric("test_loss")
        print("epoch {} train_loss={} test_loss={}".format(epoch, train_loss, test_loss))

        if test_loss < best_loss:
            best_loss = test_loss
            helper.save_confusion_model(epoch, test_loss, distinct_training_losses, distinct_validation_losses)
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement > 20:
            print("Early stopping, since no improvement in test loss")
            break

    print("Confusion model training done, best test loss={}".format(best_loss))
