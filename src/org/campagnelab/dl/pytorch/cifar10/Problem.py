import torch

from org.campagnelab.dl.pytorch.cifar10.Samplers import ProtectedSubsetRandomSampler


class Problem:
    def __init__(self, mini_batch_size=128):
        self._mini_batch_size = mini_batch_size

    def describe(self):
        print("{} Problem has {} training , {} test and {} unsupervised examples".format(
            self.name(),
            len(self.train_loader()) * self.mini_batch_size(),
            len(self.test_loader()) * self.mini_batch_size(),
            len(self.reg_loader()) * self.mini_batch_size(),
        ))

    def name(self):
        pass

    def example_size(self):
        """Returns the shape of the input, e.g., (3,32,32) for a 3 channel image with dimensions
        32x32 pixels."""
        return (0, 0, 0)

    def mini_batch_size(self):
        return self._mini_batch_size

    def train_set(self):
        """Returns the training DataSet."""
        return None

    def unsup_set(self):
        """Returns the unsupervised DataSet."""
        return None

    def test_set(self):
        """Returns the test DataSet."""
        return None

    def loader_for_dataset(self, dataset):
        pass

    def train_loader(self):
        """Returns the torch dataloader over the training set. """
        pass

    def train_loader_subset_range(self, start, end):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        return self.train_loader_subset(range(start, end))

    def train_loader_subset(self, indices):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the examples identified by these indices."""
        pass

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        pass

    def test_loader_subset(self, indices):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        pass

    def test_loader_range(self, start, end):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """
        return self.test_loader_subset(range(start,end))

    def reg_loader(self):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        pass

    def reg_loader_subset(self, indices):
        """Returns the torch dataloader over the regularization set, shuffled,
        but limited to the example range start-end."""
        pass

    def reg_loader_subset_range(self, start, end):
        """Returns the torch dataloader over the regularization set, shuffled,
        but limited to the example range start-end."""
        return self.reg_loader_subset(range(start, end))

    def loss_function(self):
        """Return the loss function for this problem."""
        pass
