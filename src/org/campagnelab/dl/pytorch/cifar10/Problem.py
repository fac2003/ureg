
class Problem:
    def __init__(self, mini_batch_size=128):
        self.mini_batch_size=mini_batch_size

    def mini_batch_size(self):
        return self.mini_batch_size

    def train_loader(self):
        """Returns the torch dataloader over the training set. """
        pass

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        pass

    def reg_loader(self):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        pass

    def loss_function(self):
        """Return the loss function for this problem."""
        pass