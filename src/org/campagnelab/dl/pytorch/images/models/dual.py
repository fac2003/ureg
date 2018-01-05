from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import Module

class LossEstimator:
    def loss(self, xs, xu):
        xs=xs.detach()
        xu=xu.detach()
        return torch.norm(xs-xu,p=1)

class Dual(Module):
    def __init__(self, loss_estimator=None, delegate_module=None):
        super(Dual).__init__()
        assert loss_estimator is not None
        assert delegate_module is not None
        self.delegate_module=delegate_module
        self.loss_estimator=loss_estimator

    def forward(self, xs, xu):
        for index, module in enumerate(self._modules.values()):
            xs = self.delegate_module(xs)
            xu = self.delegate_module(xs)
            self.loss_weight[index]*self.loss_estimator(xs,xu)
        return (xs,xu, self.loss_weight)


class SequentialDual(Module):
    """A dual sequential container. Makes it easy to calculate agreement loss between feature maps.

    """

    def __init__(self, loss_estimator, args):
        super(SequentialDual, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.loss_weight=Variable(torch.ones(len(self)),requires_grad=True)
        self.loss_estimator=loss_estimator

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def forward(self, xs, xu):
        loss_weight=self.loss_weight.clone()
        for index, module in enumerate(self._modules.values()):
            xs = module(xs)
            xu = module(xu)
            loss_weight[index]=self.loss_estimator.loss(xs,xu)
        return (xs,xu, torch.sum(loss_weight))

