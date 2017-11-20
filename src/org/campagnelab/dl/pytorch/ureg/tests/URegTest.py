import unittest

import torch
from torch.autograd import Variable
from torch.nn import Module, BCELoss, Softmax

from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer
from org.campagnelab.dl.pytorch.ureg.tests.SimpleModel import SimpleModel


class URegTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel(2, 1)
        self.criterion = BCELoss()
        self.x = Variable(torch.cat([torch.rand(1), torch.ones(1)]),requires_grad=True)
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
        regularization_loss = ureg.regularization_loss( xs, xu)
        regularization_loss.backward()

        for param in self.model.parameters():
            print("parameter grad: {}".format(param.grad.data))
            self.assertTrue(abs(param.grad.data.sum()) > self.epsilon, msg="gradient cannot be null")
            param.data.add_(-0.1 * param.grad.data)


if __name__ == '__main__':
    unittest.main()
