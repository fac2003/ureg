'''Pretrain a model using the split image strategy.'''
from __future__ import print_function

import argparse
import random
import string

import matplotlib.pyplot as plt
import numpy
import torchvision
from PIL import Image
from cv2 import cv2

from org.campagnelab.dl.pytorch.cifar10.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.cifar10.GradCam import GradCam
from org.campagnelab.dl.pytorch.cifar10.Problems import create_model
from org.campagnelab.dl.pytorch.cifar10.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.cifar10.TrainModelSplit import TrainModelSplit
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

def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(numpy.transpose(npimg, (1, 2, 0)))


def make_image(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    return Image.fromarray(npimg, mode='RGB')


def add_cam_on_image(image, mask):
    heatmap = cv2.applyColorMap(numpy.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = numpy.float32(heatmap) / 255
    heatmap=numpy.transpose(heatmap,(2,0,1))
    cam = heatmap + numpy.float32(image.numpy())
    cam = cam / numpy.max(cam)
    #npimg=numpy.uint8(255 * cam)
    return  torch.from_numpy(cam)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Evaluate split training against CIFAR10 & STL10')
    parser.add_argument('--lr', default=0.005, type=float, help='Learning rate.')
    parser.add_argument('--mini-batch-size', type=int, help='Size of the mini-batch.', default=1)
    parser.add_argument('--num-epochs', '--max-epochs', type=int,
                        help='Number of epochs to run before stopping. Additional epochs when --resume.', default=200)
    parser.add_argument('--num-classes', type=int,
                        help='Number of classes to train with for pre-training..', default=10)
    parser.add_argument('--layer', type=str, help='Name of the model layer that contains the filters.',
                        default="35")
    parser.add_argument('--momentum', type=float, help='Momentum for SGD.', default=0.9)
    parser.add_argument('--L2', type=float, help='L2 regularization.', default=1E-4)
    parser.add_argument('--checkpoint-key', help='random key to save/load checkpoint',
                        default=''.join(random.choices(string.ascii_uppercase, k=5)))
    parser.add_argument('--model', default="PreActResNet18", type=str,
                        help='The model to instantiate. One of VGG16,	ResNet18, ResNet50, ResNet101,ResNeXt29, ResNeXt29, DenseNet121, PreActResNet18, DPN92')
    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem, either CIFAR10 or STL10')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show images interactively.')

    args = parser.parse_args()

    print("Loading pre-trained model " + args.checkpoint_key)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("With CUDA")
    else:
        print("With CPU")

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

    model_trainer = TrainModelSplit(args=args, problem=problem, use_cuda=use_cuda)
    if hasattr(args, 'seed'):
        torch.manual_seed(args.seed)
        if use_cuda:
            torch.cuda.manual_seed(args.seed)
    # View image sample:

    pre_training_set = model_trainer.calculate_pre_training_set(args.num_classes, 4, shuffle=False)
    iterator = iter(pre_training_set)
    combined_set = []
    while True:
        # Assemble a grid of two half images and show it:
        try:
            image1, index1 = next(iterator)
            image2, index2 = next(iterator)
            images = torch.cat([image2, image1], dim=0)
            if args.show: imshow(torchvision.utils.make_grid(images, nrow=2, normalize=True, ))
            # the classes are the same in index1 and index2 since dataset was not shuffled:
            combined_set += [(images, index1)]
        except StopIteration:
            break

            # imshow(image1[0])

    print()

    model_trainer.init_model(create_model_function=create_model)
    model_trainer.net = model_trainer.load_pretrained_model()
    if model_trainer.net is None:
        print("Unable to load pretrained model for checkpoint key " + args.checkpoint_key)
        exit(1)

    optimizer = torch.optim.SGD(model_trainer.net.parameters(), lr=args.lr, momentum=args.momentum,
                                weight_decay=args.L2)

    criterion = problem.loss_function()

    # freeze the features part of the model:
    for param in model_trainer.net.features.parameters():
        param.requires_grad = False
    model_trainer.net.train()
    # train the classifier for a few epochs to finalize the classifier for the test images:
    for epoch in range(0, 2):
        for (batch_idx, (half_images, class_indices)) in enumerate(pre_training_set):
            class_indices = class_indices.type(torch.LongTensor)
            if use_cuda:
                half_images = half_images.cuda()
                class_indices = class_indices.cuda()

            inputs, targets = Variable(half_images), Variable(class_indices, requires_grad=False)
            optimizer.zero_grad()

            outputs = model_trainer.net(inputs)
            pre_training_loss = criterion(outputs, targets)
            pre_training_loss.backward()
            optimizer.step()
            print("classifier training loss=" + str(pre_training_loss))

    # Calculate the heatmap and add to the source images:
    grad_cam = GradCam(model=model_trainer.net, example_size=problem.example_size(),
                       target_layer_names=[args.layer], use_cuda=use_cuda)

    print("model=" + str(model_trainer.net))
    modified = []
    resizer = torchvision.transforms.Resize((224, 224))
    to_tensor = torchvision.transforms.ToTensor()
    normalizer = torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    collect_max=0
    for (images, target_index) in combined_set:
        inputs, targets = Variable(images, requires_grad=True), Variable(target_index)

        mask = grad_cam(inputs, 0)
        for img in add_cam_on_image(images, mask):
            modified += [img]
        # the heatmap indicates which part of the image was most informative to
        # arrive at the prediction of target_index for the split images.
        collect_max+=1
        if collect_max>4: break
    imshow(torchvision.utils.make_grid(modified, nrow=2, normalize=True, ))
    print()