from torch.autograd import Variable

from org.campagnelab.dl.pytorch.cifar10.AccuracyHelper import AccuracyHelper
from org.campagnelab.dl.pytorch.cifar10.LossHelper import LossHelper
from org.campagnelab.dl.pytorch.cifar10.utils import progress_bar


class TestModel:
    def __init__(self, model, use_cuda, problem):
        self.model=model
        self.use_cuda=use_cuda
        self.problem=problem
        self.criterion=problem.loss_function()
        self.mini_batch_size=problem.mini_batch_size()
        self.max_validation_examples=len(problem.test_set())
        self.num_validation=len(problem.test_set())

    def test(self, performance_estimators=(LossHelper("test_loss"), AccuracyHelper("test_"))):

        self.model.eval()
        for performance_estimator in performance_estimators:
            performance_estimator.init_performance_metrics()
        for batch_idx, (inputs, targets) in enumerate(self.problem.test_loader_range(0, self.num_validation)):

            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets, volatile=True)
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            for performance_estimator in performance_estimators:
                performance_estimator.observe_performance_metric(batch_idx, loss.data[0], outputs, targets)

            progress_bar(batch_idx * self.mini_batch_size, self.max_validation_examples,
                         " ".join([performance_estimator.progress_message() for performance_estimator in
                                   performance_estimators]))

            if ((batch_idx + 1) * self.mini_batch_size) > self.max_validation_examples:
                break
        print()

        return performance_estimators

