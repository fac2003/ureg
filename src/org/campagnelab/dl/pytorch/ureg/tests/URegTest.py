import random
import unittest

import sys
import torch
from torch.autograd import Variable
from torch.legacy.nn import MSECriterion, AbsCriterion
from torch.nn import BCELoss

from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.PerformanceList import PerformanceList
from org.campagnelab.dl.pytorch.cifar10.TrainModel import TrainModel, print_params
from org.campagnelab.dl.pytorch.ureg.URegularizer import URegularizer
from org.campagnelab.dl.pytorch.ureg.tests.DummyArgs import DummyArgs
from org.campagnelab.dl.pytorch.ureg.tests.SimpleModel import SimpleModel
import torch.nn.functional as F

from org.campagnelab.dl.pytorch.ureg.tests.TestProblem import TestProblem


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2)))


class URegTest(unittest.TestCase):
    def setUp(self):
        self.model = SimpleModel(2, 1)
        self.criterion = BCELoss()
        self.x = Variable(torch.cat([torch.rand(1), torch.ones(1)]), requires_grad=True)
        self.y = self.model.forward(self.x)
        self.y_true = Variable(torch.ones(1), requires_grad=False)
        self.epsilon = 1e-12
        self.dataset_size = 10

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
        ureg.set_num_examples(1000, 1000)

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
        random.seed(12)
        torch.manual_seed(12)
        ureg_alpha=1
        ureg = URegularizer(self.model, 1, num_features=2,
                            alpha=ureg_alpha,
                            learning_rate=0.01)
        ureg.set_num_examples(100, 100)
        training_set_bias_enabled = True
        random.seed(12)
        torch.manual_seed(12)
        self.dataset_size = 30

        train_inputs, train_targets = self.build_inputs(add_bias=training_set_bias_enabled)
        test_inputs, test_targets = self.build_inputs(add_bias=False)

        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        # model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                    weight_decay=0)

        ureg_enabled = True
        if ureg_enabled:
            ureg.enable()
        else:
            ureg.disable()

        for epoch in range(0, 62):
            average_loss = 0
            ureg.new_epoch(epoch=epoch)
            model.train()
            for index in range(0, self.dataset_size):
                optimizer.zero_grad()
                inputs, targets = Variable(train_inputs[index]), \
                                  Variable(train_targets[index], requires_grad=False)
                outputs = model(inputs)

                if ureg_enabled:

                    unsup = Variable(test_inputs[index])
                    ureg.train_ureg(inputs, unsup)
                    r_loss = ureg.regularization_loss(inputs, unsup)
                    r_loss *= ureg._alpha
                    r_loss.backward()
                    optimizer.step()
                else:
                    r_loss = 0

                supervised_loss = rmse(outputs, targets)

                average_loss += supervised_loss.data[0]
                supervised_loss.backward()
                optimizer.step()
            average_loss /= self.dataset_size
            print("epoch {} average supervised_loss= {:3f} ureg loss={:3f} acc={:3f}".format(
                epoch, average_loss,
                ureg.get_ureg_loss(),
                ureg.ureg_accuracy()))

        for param in model.parameters():
            print("parameter {}".format(param))
        print_params(0, model)

        eps = 0.001

        for index in range(0, 4):
            inputs = Variable(train_inputs[index])
            result = model(inputs)
            true_target = train_targets[index]
            print("train_inputs: ({:.3f}, {:.3f}) predicted target: {:.3f} true target: {:.1f} ".format(
                inputs.data[0, 0], inputs.data[0, 1],
                result.data[0, 0], true_target[0, 0]))

        for index in range(0, 4):
            inputs = Variable(train_inputs[index])
            result = model(inputs)
            true_target = train_targets[index]
            if abs(inputs.data[0, 1] - 0.6) < eps:
                self.assertTrue(ureg_enabled and training_set_bias_enabled and
                                result.data[0, 0] > 0.9,
                                msg="probability must be larger than 0.9 on true signal when ureg is enabled")
            print("train_inputs: ({:.3f}, {:.3f}) predicted target: {:.3f} true target: {:.1f} ".format(
                inputs.data[0, 0], inputs.data[0, 1],
                result.data[0, 0], true_target[0, 0]))

        for index in range(0, 4):
            inputs = Variable(test_inputs[index])
            result = model(inputs)
            true_target = test_targets[index]
            print("test_inputs: ({:.3f}, {:.3f}) predicted target: {:.3f} true target: {:.1f} ".format(
                inputs.data[0, 0], inputs.data[0, 1],
                result.data[0, 0], true_target[0, 0]))

    def test_optimize_with_train_model(self):
        test_problem = TestProblem()

        model_trainer = TrainModel(DummyArgs(ureg=True,
                                             optimize="similarity"), problem=test_problem, use_cuda=False)
        model_trainer.init_model(create_model_function=lambda name:

        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
                                 )

        model_trainer.ureg.set_num_examples(100, 100)

        for epoch in range(0, 100):
            estimators = PerformanceList()
            estimators += [LossHelper("train_loss")]
            estimators += [LossHelper("reg_loss")]
            model_trainer.train(epoch=epoch, performance_estimators=estimators,
                                train_supervised_model=True,
                                train_ureg=True,
                                regularize=True)
            # print("ureg loss={:3f} acc={:3f}".format(
            #    model_trainer.ureg.get_ureg_loss(),
            #    model_trainer.ureg.ureg_accuracy()))

        test_inputs = test_problem.test_loader()
        eps = 0.001
        for (index, (input, true_target)) in enumerate(test_problem.test_loader()):
            input = Variable(input)
            result = model_trainer.net(input)
            print("test_inputs: ({:.3f}, {:.3f}) predicted target: {:.3f} true target: {:.1f} ".format(
                input.data[0, 0], input.data[0, 1],
                result.data[0, 0], true_target[0, 0]))

        for (index, (input, true_target)) in enumerate(test_problem.test_loader()):
            input = Variable(input)
            result = model_trainer.net(input)
            if abs(input.data[0, 1] - 0.6) < eps:
                self.assertTrue(
                    result.data[0, 0] > 0.8,
                    msg="probability must be larger than 0.9 on true signal when ureg is enabled")

            if abs(input.data[0, 0] - 0.45) < eps and abs(input.data[0, 1] - 0.4) < eps:
                self.assertTrue(
                    result.data[0, 0] < 0.6,
                    msg="probability must be larger than 0.4 on biased signal when ureg is enabled")

    def test_optimize_with_train_model_two_passes(self):
        test_problem = TestProblem()

        model_trainer = TrainModel(DummyArgs(ureg=True,
                                             lr=0.001,
                                             shave_lr=0.01), problem=test_problem, use_cuda=False)
        model_trainer.init_model(create_model_function=lambda name:

        torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
                                 )

        model_trainer.ureg.set_num_examples(100, 100)


        for epoch in range(0, 100):
            estimators=PerformanceList()
            estimators+=[LossHelper("train_loss")]
            estimators += [LossHelper("reg_loss")]

            model_trainer.train(epoch=epoch, performance_estimators=estimators,
                                train_supervised_model=True,
                                train_ureg=True,
                                regularize=False)
            model_trainer.regularize(epoch=epoch)
            print("train_loss {:3f} ureg loss={:3f}".format(
                estimators.get_metric("train_loss"),
                estimators.get_metric("reg_loss")))

        test_inputs = test_problem.test_loader()
        eps = 0.001
        print("\n")

        for (index, (input, true_target)) in enumerate(test_problem.test_loader()):
            input = Variable(input)
            result = model_trainer.net(input)
            print("test_inputs: ({:.3f}, {:.3f}) predicted target: {:.3f} true target: {:.1f} ".format(
                input.data[0, 0], input.data[0, 1],
                         result.data[0, 0], true_target[0, 0]))
        sys.stdout.flush()
        for (index, (input, true_target)) in enumerate(test_problem.test_loader()):
            input = Variable(input)
            result = model_trainer.net(input)
            if abs(input.data[0, 1] - 0.6) < eps:
                self.assertTrue(
                    result.data[0, 0] > 0.8,
                    msg="probability must be larger than 0.9 on true signal when ureg is enabled")

            if abs(input.data[0, 0] - 0.45) < eps and abs(input.data[0, 1] - 0.4) < eps:
                self.assertTrue(
                    result.data[0, 0] < 0.6,
                    msg="probability must be larger than 0.4 on biased signal when ureg is enabled")


    def build_inputs(self, add_bias):
        bs = self.dataset_size
        inputs = [[0, 0]] * bs
        targets = [[0]] * bs

        for index in range(0, bs):
            a = 0.45 if random.uniform(0., 1.) > 0.6 else 0.5
            b = 0.6 if random.uniform(0., 1.) > 0.5 else 0.4

            if b > 0.5:
                y = 1
            else:
                y = 0
            if add_bias and y == 0:
                # add a few spurious associations:
                if a < 0.5:
                    y = 1
            value = torch.FloatTensor([[a, b]])

            target = torch.FloatTensor([[y]])
            print("inputs=({:.3f},{:.3f}) targets={:.1f}".format(value[0, 0], value[0, 1], target[0, 0]))

            inputs[index] = value
            targets[index] = target
        return inputs, targets


if __name__ == '__main__':
    unittest.main()
