from collections import OrderedDict

import torch
from torch.autograd import Variable
from torch.nn import Module


def LossEstimator_L1_cpu(xs, xu):
    xs=xs.detach()
    xu=xu.detach()
    loss = 0
    bs = xs.size()[0]
    is_cuda=xs.is_cuda
    abs = torch.abs(xs - xu)
    xs = xs.cpu()
    xu = xu.cpu()
    abs=abs.cpu()
    n = 0
    for index_s in range(bs):
        norm = abs[index_s].norm(p=1)
        for index_u in range(bs):
            if index_u>=index_s:
                # squared_norm = norm * norm
                # regularize so that the features are either similar or orthogonal:
                dot_product = xs[index_s].dot(xu[index_u])
                loss = loss + torch.min(norm, dot_product)
                n += 1
    value=loss / n
    return value.cuda() if is_cuda else value

def LossEstimator_L1( xs, xu):
        xs=xs.detach()
        xu=xu.detach()
        loss_variable = Variable(torch.zeros(1), requires_grad=True)
        if xs.is_cuda:
            loss_variable = loss_variable.cuda()
        bs=xs.size()[0]
        xs=xs.view(bs,-1)
        xu=xu.view(bs,-1)

        n=0
        for index_s in range(bs):
            # expand x1 by copying values  in a batch-size by batch-size matrix of length num features.
            x1 = xs[index_s].expand(bs, 1, -1)
            # regularize so that the features are either similar or orthogonal:
            dot_product = x1.bmm(xu.view(bs,-1,1))
            for index_u in range(bs):
                if index_u>=index_s:
                    norm = (xs[index_s]-xu[index_u]).norm(p=1)
                    #if bs>1:
                    #    print("STOP")
                    dot_prod = dot_product.view(bs,-1)[index_u]
                    #if bs > 1:
                    #    print("is={} iu={} norm={} dot_prod={}".format(index_s, index_u, norm.data[0], dot_prod.data[0]))
                    loss_variable = loss_variable + torch.min(norm, dot_prod)
                    n+=1
        return loss_variable / n

class Dual(Module):
    def __init__(self, loss_estimator=None, delegate_module=None):
        super().__init__()
        assert loss_estimator is not None
        assert delegate_module is not None
        self.delegate_module=delegate_module
        self.loss_estimator=loss_estimator


    def forward(self, xs, xu):
        loss_weight=0.0
        for index, module in enumerate(self._modules.values()):
            xs = self.delegate_module(xs)
            if xu is not None:
                xu = self.delegate_module(xu)
                loss_weight=self.loss_estimator(xs,xu)
        return (xs,xu, loss_weight)

    def cuda(self, device=None):
        self.delegate_module.cuda(device)

class SequentialDual(Module):
    """A dual sequential container. Makes it easy to calculate agreement loss between feature maps.
    """

    def __init__(self, loss_estimator,  args):
        """
        :param loss_estimator: a function f(xs,xu) that returns the loss for agreement between xs and xu.
        :param args:
        """
        super(SequentialDual, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        self.loss_weight=Variable(torch.zeros(1),requires_grad=True)
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
        loss_weight=0.0


        for index, module in enumerate(self._modules.values()):
            xs = module(xs)
            if xu is not None:
                xu = module(xu)
                loss_weight+=self.loss_estimator(xs,xu)
        return (xs,xu, loss_weight)


if __name__ == '__main__':
    l_cpu = LossEstimator_L1_cpu
    l_gpu = LossEstimator_L1

    x= Variable(torch.ones(4,10), requires_grad=True)
    colinear_loss= l_cpu(x, x*2)
    answer=10
    assert colinear_loss.data[0]==answer, "colinear loss {} must match {}".format(colinear_loss.data[0],answer)
    a = Variable(torch.rand(10, 100), requires_grad=True)
    b = Variable(torch.rand(10, 100), requires_grad=True)

    data_cpu = l_cpu(a, b).data[0]
    data_gpu = l_gpu(a, b).data[0]
    assert data_cpu == data_gpu, "loss must match when calculated on cpu {} or GPU {}: ".format(data_cpu,data_gpu)
