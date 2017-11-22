
class PerformanceEstimator:
    def init_performance_metrics(self):
        """Initializes accumulators. Must be called before starting another round of estimations.
        """
        pass

    def observe_performance_metric(self, iteration, loss, outputs, targets):
        """Collect information to calculate metrics. Must be called at each iteration (mini-batch).
        """
        pass

    def metric_names(self):
        """
        Names of estimated metrics.
        :return: list of names of evaluated metrics.
        """
        pass

    def estimates_of_metric(self):
        """
        Return a list of metric estimates.
        :return: List of float values for evaluated metrics.
        """
        pass

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        pass