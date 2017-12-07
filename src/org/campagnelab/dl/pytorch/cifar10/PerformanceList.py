class PerformanceList(list):


    def __init__(self) -> None:
        super().__init__()
        #self.list=[]

   # def append(self, values):
   #     self.list.append(values)

    def get_metric(self, metric_name):
        for pe in self:
            metric = pe.get_metric(metric_name)
            if metric is not None:
                return metric
        return None

    def set_metric(self, iteration, metric_name, value):
        for pe in self:
            pe.observe_named_metric(iteration, metric_name, value, None, None)

    def set_metric_with_outputs(self, iteration, metric_name, loss, outputs, targets):
        for pe in self:
            pe.observe_named_metric(iteration, metric_name, loss, outputs, targets)

    def init_performance_metrics(self):
        for pe in self:
            pe.init_performance_metrics()
