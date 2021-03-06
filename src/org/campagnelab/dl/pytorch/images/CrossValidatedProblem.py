from org.campagnelab.dl.pytorch.images.Problem import Problem


class CrossValidatedProblem(Problem):
    """Wraps another problem and exposes a training and test set constructed as follows:
        Both training and test sets will be made up of subsets of the original problem's training
        set. The examples returned are selected using training_indices and validation_indices.
        These set of indices should be disjoint.
        The unsupervised set is returned unchanged from the delegate problem.
    """


    def __init__(self, delegate, training_indices, validation_indices=None):
        """
        Construct a cross-validated problem.
        :param delegate: original problem to be wrapped.
        :param training_indices: indices of the training samples within the original problem's training set.
        :param validation_indices: indices of the validation samples within the original problem's training set.
        :param mini_batch_size: size of the mini batch.
        """
        super().__init__(delegate.mini_batch_size())
        self.delegate = delegate
        self.training_indices = training_indices
        if validation_indices is None:
            # calculate the complement: indices in training set of original problem, not in training indices:
            all = range(0, len(self.delegate.train_set()))
            complement = set(all) - set(training_indices)
            validation_indices = [i for i in complement]
        self.validation_indices = validation_indices


    def loader_for_dataset(self, dataset):
        return self.delegate.loader_for_dataset(dataset)

    def train_set(self):
        return [self.delegate.train_set()[index] for index in range(0,len(self.training_indices))]

    def delegate_train_index(self, cv_index):
        """Return the index of the training sample in the delegate."""
        return self.training_indices[cv_index]

    def delegate_validation_index(self, cv_index):
        """Return the index of the validation sample in the delegate."""
        return self.validation_indices[cv_index]

    def unsup_set(self):

        return self.delegate.unsup_set()

    def test_set(self):
        return [self.delegate.train_set()[index] for index in range(0, len(self.validation_indices))]

    def name(self):
        return "cross-validated " + self.delegate.name()

    def reg_loader(self):
        return self.delegate.reg_loader()

    def reg_loader_subset(self, indices):
        return self.delegate.reg_loader_subset(indices)

    def train_loader(self):
        return self.delegate.train_loader_subset(self.training_indices)

    def test_loader(self):
        return self.delegate.train_loader_subset(self.validation_indices)

    def train_loader_subset(self, indices):
        # find corresponding indices in the delegate wrapper:
        delegate_indices = [self.training_indices[index] for index in indices]
        return self.delegate.train_loader_subset(delegate_indices)

    def test_loader_subset(self, indices):
        # find corresponding indices in the delegate wrapper:
        delegate_indices = [self.validation_indices[index] for index in indices]
        return self.delegate.train_loader_subset(delegate_indices)

    def example_size(self):
        """Returns the shape of the input, e.g., (3,32,32) for a 3 channel image with dimensions
        32x32 pixels."""
        return self.delegate.example_size()

    def mini_batch_size(self):
        return self.delegate.mini_batch_size()

    def loss_function(self):
        return self.delegate.loss_function()

    def num_classes(self):
        return self.delegate.num_classes()

