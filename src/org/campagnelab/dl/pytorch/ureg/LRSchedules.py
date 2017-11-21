# learning rate schedules that help train ureg regularized models.
# The trick is to reset the learning rate to a high value periodically (learning rate annealing) to allow the optimizer
# to explore the shaved loss space.
from torch.optim.lr_scheduler import _LRScheduler, ExponentialLR, LambdaLR, ReduceLROnPlateau


class LearningRateAnnealing(ReduceLROnPlateau):

    def __init__(self, optimizer, anneal_every_n_epoch=None, delegate=None):
        assert anneal_every_n_epoch is not None, "anneal_every_n_epoch must be defined"
        self.current_epoch = 0
        self.anneal_every_n_epoch = anneal_every_n_epoch
        if delegate is None:
            delegate = ExponentialLR(optimizer, gamma=0.5)
        # the     resetLR will reset the learning rate to its initial value on each step we call it:
        self.resetLR = LambdaLR(optimizer, lr_lambda=lambda epoch: 1)
        self.delegate = delegate
        super().__init__(optimizer)


    def step(self, metrics, epoch=None):
        self.current_epoch=self.current_epoch+1
        if (self.current_epoch>self.anneal_every_n_epoch):
            # we have reached the number of epoch to anneal:
            self.resetLR.step(epoch)
            self.current_epoch = 0
        else:
            # simply decay according to the other delegate:
            self.delegate.step(metrics, epoch)