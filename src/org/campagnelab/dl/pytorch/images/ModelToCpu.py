'''Load a CUDA model and save it back for the CPU.'''
from __future__ import print_function

import argparse

from org.campagnelab.dl.pytorch.images.Cifar10Problem import Cifar10Problem
from org.campagnelab.dl.pytorch.images.STL10Problem import STL10Problem
from org.campagnelab.dl.pytorch.images.TrainModelSplit import TrainModelSplit
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

    parser = argparse.ArgumentParser(description='Load a CUDA model and save it back for the CPU.')

    parser.add_argument('--checkpoint-key', help='key to load and save the checkpoint model.', type=str)
    parser.add_argument('--problem', default="CIFAR10", type=str,
                        help='The problem, either CIFAR10 or STL10')
    args = parser.parse_args()
    if args.checkpoint_key is None:
        print("You must specify a checkpoint key.")
        exit(1)

    print("Loading pre-trained model " + args.checkpoint_key)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("With CUDA")
    else:
        print("With CPU")

    if args.problem == "CIFAR10":
        problem = Cifar10Problem(1)
    elif args.problem == "STL10":
        problem = STL10Problem(1)
    else:
        print("Unsupported problem: " + args.problem)
        exit(1)

    model_trainer = TrainModelSplit(args=args, problem=problem, use_cuda=use_cuda)
    model_trainer.net=model_trainer.load_pretrained_model()
    model_trainer.net.cpu()
    model_trainer.save_pretrained_model()
    print("Model converted to CPU.")