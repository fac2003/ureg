
class Problem:
    def __init__(self, mini_batch_size=128):
        self._mini_batch_size=mini_batch_size

    def mini_batch_size(self):
        return self._mini_batch_size

    def train_loader(self):
        """Returns the torch dataloader over the training set. """
        pass

    def train_loader_subset(self, start, end):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        pass

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        pass

    def reg_loader(self):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        pass

    def reg_loader_subset(self, start, end):
        """Returns the torch dataloader over the regularization set, shuffled,
        but limited to the example range start-end."""
        pass
    def loss_function(self):
        """Return the loss function for this problem."""
        pass