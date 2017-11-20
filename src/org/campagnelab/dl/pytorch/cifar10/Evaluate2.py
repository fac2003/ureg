'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

# MIT License
#
# Copyright (c) 2017 liukuang
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


import argparse
import os
import random
import string

import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau

from org.campagnelab.dl.pytorch.cifar10.models import *
from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar

parser = argparse.ArgumentParser(description='Evaluate ureg against CIFAR10')
parser.add_argument('--lr', default=0.005, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--ureg', '-u', action='store_true', help='Enable unsupervised regularization (ureg)')
parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch', default=128)
parser.add_argument('--num-training', '-n', type=int, help='Maximum number of training examples to use',
                    default=sys.maxsize)
parser.add_argument('--num-validation', '-x', type=int, help='Maximum number of training examples to use',
                    default=sys.maxsize)
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


args = parser.parse_args()

print("Executing " + args.checkpoint_key)

use_cuda = torch.cuda.is_available()
is_parallel = False
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

mini_batch_size = args.mini_batch_size
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=2)

# transform the unsupervised set the same way as the training set:
unsupset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
unsuploader = torch.utils.data.DataLoader(unsupset, batch_size=mini_batch_size, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    ureg_enabled = checkpoint['ureg']

    if ureg_enabled:
        ureg = URegularizer(net, mini_batch_size, args.ureg_num_features,
                            args.ureg_alpha, args.ureg_learning_rate)
        ureg.set_num_examples(args.num_training, len(unsuploader))
        ureg.enable()
        ureg.resume(checkpoint['ureg_model'])
else:
    print('==> Building model {}'.format(args.model))


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

    function = models[args.model]
    if function is None:
        print("Wrong model name: " + args.model)
        exit(1)
    # construct the model specified on the command line:
    net = function()
    ureg_enabled = args.ureg

if use_cuda:
    net.cuda()
# net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer_training = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
optimizer_reg = optim.SGD(net.parameters(), lr=args.shave_lr, momentum=0.9, weight_decay=5e-4)
ureg = URegularizer(net, mini_batch_size, num_features=args.ureg_num_features,
                    alpha=args.ureg_alpha,
                    learning_rate=args.ureg_learning_rate)
if args.ureg:
    ureg.enable()
    ureg.set_num_examples(args.num_training, len(unsuploader))
    ureg.forget_model(args.ureg_reset_every_n_epoch)
    print(
        "ureg is enabled with alpha={}, reset every {} epochs. ".format(args.ureg_alpha, args.ureg_reset_every_n_epoch))

else:
    ureg.disable()
    ureg.set_num_examples(args.num_training, len(unsuploader))
    print("ureg is disabled")

scheduler_train = ReduceLROnPlateau(optimizer_training, 'min', factor=0.5, patience=args.lr_patience, verbose=True)
scheduler_reg = ReduceLROnPlateau(optimizer_reg, 'min', factor=0.5, patience=args.lr_patience, verbose=True)
max_training_examples = args.num_training
max_validation_examples = args.num_validation

unsupiter = iter(unsuploader)

metrics = ["epoch", "checkpoint", "training_loss", "training_accuracy", "test_accuracy", "supervised_loss", "test_loss",
           "unsupervised_loss", "delta_loss", "ureg_accuracy", "ureg_alpha"]

if not args.resume:
    with open("all-perfs-{}.tsv".format(args.checkpoint_key), "w") as perf_file:
        perf_file.write("\t".join(map(str, metrics)))
        perf_file.write("\n")

    with open("best-perf-{}.tsv".format(args.checkpoint_key), "w") as perf_file:
        perf_file.write("\t".join(map(str, metrics)))
        perf_file.write("\n")


def format_nice(n):
    try:
        if n == int(n):
            return str(n)
        if n == float(n):
            return "{0:.4f}".format(n)
    except:
        return str(n)


best_test_loss = 100


def log_performance_metrics(epoch, training_loss, supervised_loss, training_accuracy, unsupervised_loss,
                            test_loss, test_accuracy, ureg_accuracy, alpha):
    global best_acc
    delta_loss = test_loss - supervised_loss
    metrics = [epoch, args.checkpoint_key, training_loss, training_accuracy, test_accuracy,
               supervised_loss, test_loss,
               unsupervised_loss, delta_loss, ureg_accuracy, alpha]
    with open("all-perfs-{}.tsv".format(args.checkpoint_key), "a") as perf_file:
        perf_file.write("\t".join(map(format_nice, metrics)))
        perf_file.write("\n")

    if test_accuracy>= best_acc:
         with open("best-perf-{}.tsv".format( args.checkpoint_key), "a") as perf_file:
             perf_file.write("\t".join(map(format_nice, metrics)))
             perf_file.write("\n")


# Training


def train(epoch, unsupiter):
    print('\nTraining, epoch: %d' % epoch)
    net.train()
    average_total_loss = 0
    average_supervised_loss = 0
    average_unsupervised_loss = 0
    correct = 0
    total = 0
    average_total_loss = 0
    unsupervised_loss = 0
    training_accuracy = 0
    supervised_loss = 0
    optimized_loss = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer_training.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        supervised_loss = criterion(outputs, targets)
        supervised_loss.backward()
        optimizer_training.step()
        # the unsupervised regularization part goes here:
        try:
            # first, read a minibatch from the unsupervised dataset:
            ufeatures, ulabels = next(unsupiter)

        except StopIteration:
            unsupiter = iter(unsuploader)
            ufeatures, ulabels = next(unsupiter)
        if use_cuda: ufeatures = ufeatures.cuda()
        # then use it to calculate the unsupervised regularization contribution to the loss:
        uinputs = Variable(ufeatures)

        ureg.train_ureg(inputs, uinputs)

        optimized_loss = supervised_loss
        average_total_loss += optimized_loss.data[0]
        average_supervised_loss += supervised_loss.data[0]

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        denominator = batch_idx + 1
        average_total_loss = average_total_loss / denominator
        average_supervised_loss = average_supervised_loss / denominator
        average_unsupervised_loss = average_unsupervised_loss / denominator
        training_accuracy = 100. * correct / total
        progress_bar(batch_idx, len(trainloader),
                     ('loss: %.3f s: %.3f u: %.3f | Acc: %.3f%% (%d/%d)'
                     % (average_total_loss,
                        average_supervised_loss,
                        average_unsupervised_loss,
                        training_accuracy,
                        correct,
                        total)))
        if (batch_idx + 1) * mini_batch_size > max_training_examples:
            break

    print()
    return (average_total_loss, average_supervised_loss, training_accuracy)


def regularize(epoch, unsupiter):
    print('\nRegularizing, epoch: %d' % epoch)
    net.train()
    average_total_loss = 0
    supervised_loss = 0
    trainiter=iter(trainloader)
    train_examples_used=0
    for shaving_index in range(args.shaving_epochs):
        print("Shaving step {}".format(shaving_index))
        average_unsupervised_loss = 0
        denominator=0
        for batch_idx, (inputs, targets) in enumerate(unsuploader):

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            optimizer_reg.zero_grad()
            uinputs, _ = Variable(inputs), Variable(targets)

            # don't use more training examples than allowed (-n) even if we don't use
            # their labels:
            if train_examples_used>args.num_training:
                trainiter = iter(trainloader)
                train_examples_used=0
            try:
                # first, read a minibatch from the unsupervised dataset:
                features, ulabels = next(trainiter)

            except StopIteration:
                trainiter = iter(trainloader)
                features, _ = next(trainiter)
            train_examples_used += 1
            if use_cuda: features = features.cuda()

            # then use it to calculate the unsupervised regularization contribution to the loss:
            inputs = Variable(features)
            regularization_loss = ureg.regularization_loss(inputs, uinputs)
            if regularization_loss is not None:

                regularization_loss.backward()
                optimizer_reg.step()
                optimized_loss = regularization_loss
                average_total_loss += optimized_loss.data[0]
            else:
                optimized_loss = supervised_loss

            average_unsupervised_loss += (0 if regularization_loss is None
                                          else  regularization_loss.data[0])

            denominator = batch_idx + 1
            average_total_loss = average_total_loss / denominator
            average_unsupervised_loss = average_unsupervised_loss / denominator

            progress_bar(batch_idx, len(trainloader), ' u: %.3f'
                         % (average_unsupervised_loss,))
            print()

    return (average_unsupervised_loss,)


def test(epoch):
    global best_acc
    net.eval()
    test_loss_accumulator = 0
    correct = 0
    total = 0
    test_accuracy = None
    test_loss = None
    ureg.new_epoch(epoch)
    for batch_idx, (inputs, targets) in enumerate(testloader):

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss_accumulator += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        ureg.estimate_accuracy(inputs)
        test_accuracy = 100. * correct / total
        test_loss = test_loss_accumulator / (batch_idx + 1)
        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss, test_accuracy, correct, total))

        if ((batch_idx + 1) * mini_batch_size) > max_validation_examples:
            break
    print()

    # Apply learning rate schedule:
    scheduler_train.step(test_loss, epoch=epoch)
    scheduler_reg.step(test_loss, epoch=epoch)
    ureg.schedule(test_loss, epoch)
    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.module if is_parallel else net,
            'acc': acc,
            'epoch': epoch,
            'ureg': ureg_enabled,
            'ureg_model': ureg._which_one_model
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt_{}.t7'.format(args.checkpoint_key))
        best_acc = acc

    return (test_loss, test_accuracy, ureg.ureg_accuracy(), ureg._alpha)


for epoch in range(start_epoch, start_epoch + 200):
    perfs = train(epoch, unsupiter)
    if (args.ureg):
        perfs += regularize(epoch, unsupiter)
    else:
        perfs += (float('nan'))
    perfs += test(epoch)
    log_performance_metrics(epoch, *perfs)
