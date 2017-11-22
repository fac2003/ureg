import torch

from org.campagnelab.dl.pytorch.cifar10.PerformanceEstimator import PerformanceEstimator


class AccuracyHelper(PerformanceEstimator):
    def __init__(self):
        self.init_performance_metrics()

    def init_performance_metrics(self):

        self.total = 0
        self.correct = 0
        self.accuracy = 0

    def estimates_of_metric(self):
        accuracy = 100. * self.correct / self.total
        return [accuracy, self.correct, self.total]

    def metric_names(self):
        return ["accuracy", "correct", "total"]

    def observe_performance_metric(self, iteration, loss, outputs, targets):
        _, predicted = torch.max(outputs.data, 1)
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).cpu().sum()

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        "Acc: {:.4f} {}/{}".format(*self.estimates_of_metric())