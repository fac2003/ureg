import random

import torch
from torch.utils.data import DataLoader

from org.campagnelab.dl.pytorch.cifar10.Problem import Problem


class ProblemLoader(DataLoader):

    def __init__(self, example_list):
        self.list=example_list

    def __len__(self):
        return len(self.list)


    def __iter__(self):
        return iter(self.list)

def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2)))


class TestProblem(Problem):
    def __init__(self, mini_batch_size=1):
        super().__init__(mini_batch_size)
        random.seed(12)
        torch.manual_seed(12)

        self.biased_inputs=self.build_inputs(add_bias=True)
        random.seed(12)
        torch.manual_seed(12)

        self.nobias_inputs=self.build_inputs(add_bias=False)

    def train_loader(self):
        return ProblemLoader(self.biased_inputs)

    def test_loader(self):
        return ProblemLoader(self.nobias_inputs)

    def reg_loader(self):
        return ProblemLoader(self.nobias_inputs)

    def reg_loader_subset(self, indices):
        return iter(ProblemLoader(self.nobias_inputs))

    def train_loader_subset(self, indices):
        return iter(ProblemLoader(self.biased_inputs))

    def test_loader_subset(self, indices):
        return iter(ProblemLoader(self.nobias_inputs))

    def loss_function(self):
        return rmse

    def build_inputs(self, add_bias, dataset_size=10):
        combined=[(0,0)]*dataset_size
        for index in range(0, dataset_size):
            a = 0.45 if random.uniform(0., 1.) > 0.5 else 0.5
            b = 0.6 if random.uniform(0., 1.) > 0.5 else 0.4
            value = torch.FloatTensor([[a, b]])

            if b > 0.5:
                y = 1.
            else:
                y = 0.
            if add_bias and y == 0:
                # add a few spurious associations:
                if a < 0.5:
                    y = 1.

            target = torch.FloatTensor([[y]])
            print("inputs=({:.3f},{:.3f}) targets={:.1f}".format(value[0, 0], value[0, 1], target[0, 0]))


            combined[index]=(value,target)
        return combined

