import random

import math

import torch
from torch.autograd import Variable


def get_random_slope():
    pi = math.atan(1) * 4
    angle = random.uniform(pi, -pi)
    slope = math.tan(angle)
    return slope


def half_images(inputs, slope, cuda=False):
    def above_line(xp, yp, slope, b):
        # remember that y coordinate increase downward in images
        yl = slope * xp + b
        return yp < yl

    def on_line(xp, yp, slope, b):
        # remember that y coordinate increase downward in images
        yl = slope * xp + b
        return abs(yp - yl) < 2

    uinputs = Variable(inputs, requires_grad=True)
    uinputs2 = Variable(inputs, requires_grad=False)

    mask_up = torch.ByteTensor(uinputs.size()[2:])  # index 2 drops minibatch size and channel dimension.
    mask_down = torch.ByteTensor(uinputs.size()[2:])  # index 2 drops minibatch size and channel dimension.

    channels = uinputs.size()[1]
    width = uinputs.size()[2]
    height = uinputs.size()[3]
    # print("mask down-------------")
    # fill in the first channel:
    for x in range(0, width):
        for y in range(0, height):
            above_the_line = above_line(x - width / 2, y, slope=slope, b=height / 2.0)
            on_the_line = on_line(x - width / 2, y, slope=slope, b=height / 2.0)
            mask_up[x, y] = 1 if above_the_line and not on_the_line else 0
            mask_down[x, y] = 0 if above_the_line  else \
                (1 if not on_the_line else 0)
            # print("." if mask_down[x, y] else " ",end="")
            #    print("." if mask_up[x, y] else " ",end="")
    # print("|")

    # print("-----------mask down")
    mask1 = torch.ByteTensor(uinputs.size()[1:])
    mask2 = torch.ByteTensor(uinputs.size()[1:])

    if cuda:
        mask_up = mask_up.cuda()
        mask_down = mask_down.cuda()
        mask1 = mask1.cuda()
        mask2 = mask2.cuda()
    for c in range(0, channels):
        mask1[c] = mask_up
        mask2[c] = mask_down
    mask1 = Variable(mask1, requires_grad=False)
    mask2 = Variable(mask2, requires_grad=False)

    return uinputs.masked_fill(mask1, 0), uinputs2.masked_fill(mask2, 0), mask2
