
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
        return []

    def estimates_of_metric(self):
        """
        Return a list of metric estimates.
        :return: List of float values for evaluated metrics.
        """
        return []

    def progress_message(self):
        """ Return a message suitable for logging progress of the metrics."""
        ""

    def get_metric(self,metric_name):
        """ return the metric estimate corresponding to this name, or None if not estimated
        by this estimator."""
        for index in range(len(self.metric_names())):
            if self.metric_names()==metric_name:
                return self.estimates_of_metric()[index]
        return None