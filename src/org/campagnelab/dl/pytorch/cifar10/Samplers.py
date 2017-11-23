import torch
from collections import Iterator
from torch.utils.data.sampler import Sampler
from itertools import islice


class TrimSampler(Sampler):
    """Samples elements sequentially, always in the same order, within the provided index bounds.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, start=0, end=None):
        if end is None:
            end = len(data_source)
        self.start = max(0, start)
        self.end = min(end, len(data_source))

        self.data_source = data_source

    def __iter__(self):
        return islice(iter(self.data_source), self.start, self.end)

    def __len__(self):
        return self.end - self.start


class ProtectedIterator(Iterator):
    def __init__(self, max_length, it):
        self.counter = 0
        self.max_length = max_length
        self.delegate = it

    def __next__(self):
        if self.counter < self.max_length:
            self.counter += 1
            return next(self.delegate)
        else:
            raise StopIteration()


class ProtectedSubsetRandomSampler(Sampler):
    """Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (list): a list of indices
    """

    def __init__(self, indices):
        self.indices = indices

        self.length = len(indices)

    def __iter__(self):
        return ProtectedIterator(self.length,
                                 it=iter([self.indices[i] for i in torch.randperm(len(self.indices))]))

    def __len__(self):
        return len(self.indices)
