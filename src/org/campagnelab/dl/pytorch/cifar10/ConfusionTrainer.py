'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import argparse
import collections
import random

from torch.nn import CrossEntropyLoss

from org.campagnelab.dl.pytorch.cifar10.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.cifar10.Problems import create_model
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem
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


def class_label(num_classes, predicted_index, true_index):
    return predicted_index*num_classes + true_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a confusion model.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate.')
    parser.add_argument('--mini_batch_size', default=100, type=int, help='Size of the mini-batch.')
    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem that confusion data was collected for, either CIFAR10 or STL10')
    parser.add_argument('--confusion-data', type=str,
                        help='File with confusion data to train the model with.')
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to train before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')

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
    Confusion = collections.namedtuple('Confusion', 'trained_with example_index epoch train_loss predicted_label true_label')

    confusion_data = []
    with open(args.confusion_data, mode="r") as conf_data:
        for line in conf_data.readlines():
                line=line.replace("\n","")
                trained_with, epoch, example_index, train_loss, predicted_label, true_label = line.split("\t")
                true_label=true_label.split("\n")[0]
                confusion_data += [Confusion(bool(trained_with), int(example_index), int(epoch), \
                                             float(train_loss), int(predicted_label), int(true_label))]

    model = ConfusionModel(create_model(args.model, problem),problem)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.L2)
    criterion=CrossEntropyLoss()
    random.shuffle(confusion_data)
    for epoch in range(0, args.num_epochs):

        for batch_idx, confusion_list in enumerate(batch(confusion_data, args.mini_batch_size)):
            images=[None]*args.mini_batch_size
            targets=torch.zeros(args.mini_batch_size)
            optimizer.zero_grad()
            training_loss_input = torch.zeros(args.mini_batch_size, 1)

            trained_with_input =torch.zeros(args.mini_batch_size, 1)

            for index, confusion in enumerate(confusion_list):

                num_classes = problem.num_classes()
                targets[index]=class_label(num_classes,
                                           confusion.predicted_label,confusion.true_label)
                dataset=problem.train_set() if confusion.trained_with  else problem.test_set()
                images[index], _ = dataset[confusion.example_index]
                training_loss_input[index]=confusion.train_loss
                training_loss_input[index]=1.0 if confusion.trained_with else 0.0

            image_input=Variable(torch.stack(images,dim=0), requires_grad=True)
            training_loss_input=Variable(training_loss_input, requires_grad=True)
            trained_with_input=Variable(trained_with_input, requires_grad=True)
            targets=Variable(targets, requires_grad=False).type(torch.LongTensor)

            outputs=model( training_loss_input, trained_with_input,image_input)
            loss=criterion(outputs,targets)
            print("training loss: "+str(loss.data[0]))
            loss.backward()
            optimizer.step()