import random
import unittest

import numpy
import torch
from torch.autograd import Variable
from torch.nn import Module, BCELoss, Softmax, Linear, Sequential

from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer
from org.campagnelab.dl.pytorch.ureg.tests.SimpleModel import SimpleModel


class URegTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel(2, 1)
        self.criterion = BCELoss()
        self.x = Variable(torch.cat([torch.rand(1), torch.ones(1)]), requires_grad=True)
        self.y = self.model.forward(self.x)
        self.y_true = Variable(torch.ones(1), requires_grad=False)
        self.epsilon = 1e-12

    def test_grad_not_null(self):
        """
                Checks that the gradient flowing from a simple model is not zero.

        """

        loss = self.criterion(self.y, self.y_true)
        loss.backward()

        for param in self.model.parameters():
            print("parameter grad: {}".format(param.grad.data))

            self.assertTrue(abs(param.grad.data.sum()) > self.epsilon, msg="gradient cannot be null")
            param.data.add_(-0.1 * param.grad.data)

    def test_grad_not_null_from_ureg(self):
        """
        Checks that the gradient flowing from ureg is not zero.
        """
        ureg = URegularizer(self.model, 1, num_features=2,
                            alpha=1,  # gradient only from ureg.
                            learning_rate=0.1)

        loss = self.criterion(self.y, self.y_true)
        xs = self.x
        xu = Variable(self.x.data + torch.ones(2))
        ureg.train_ureg(xs, xu)
        regularization_loss = ureg.regularization_loss(xs, xu)
        regularization_loss.backward()

        for param in self.model.parameters():
            print("parameter grad: {}".format(param.grad.data))
            self.assertTrue(abs(param.grad.data.sum()) > self.epsilon, msg="gradient cannot be null")
            param.data.add_(-0.1 * param.grad.data)

    def test_optimize_with_ureg(self):
        """
           Check that training a model with ureg can prevent overfitting to the training set.
        """
        ureg = URegularizer(self.model, 1, num_features=2,
                            alpha=1,
                            learning_rate=0.001)
        ureg.set_num_examples(100, 100)
        train_inputs, train_targets = self.build_inputs(add_bias=True)
        test_inputs, test_targets = self.build_inputs(add_bias=False)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9,
                                    weight_decay=0.0)
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Softmax())
        for epoch in range(0, 10):
            for index in range(0, 100):
                optimizer.zero_grad()
                inputs, targets = Variable(train_inputs[index]), \
                                  Variable(train_targets[index], requires_grad=False)
                outputs = model(inputs)

                unsup = Variable(test_inputs[index])
                ureg.train_ureg(inputs, unsup)

                r_loss = ureg.regularization_loss(inputs, unsup)
                r_loss.backward()
                optimizer.step()

                supervised_loss = self.criterion(outputs, targets)
                supervised_loss.backward()
                optimizer.step()

        for param in model.parameters():
            print("parameter {}".format(param))

    def build_inputs(self, add_bias):
        inputs = [[0, 0]] * 100
        targets = [[0, 0]] * 100

        for index in range(0, 100):
            a = random.uniform(0.4, 1.)
            b = random.uniform(0., 1.)
            value = torch.FloatTensor([[a, b]])

            if b > 0.5:
                y = 1
            else:
                y = 0
            if add_bias and y == 0:
                # add a few spurious associations:
                if a < 0.5:
                    y = 1

            target = torch.FloatTensor([[1., 0.]]) if y == 1 else torch.FloatTensor([[0., 1.]])

            inputs[index] = value
            targets[index] = target

        return inputs, targets


if __name__ == '__main__':
    unittest.main()
