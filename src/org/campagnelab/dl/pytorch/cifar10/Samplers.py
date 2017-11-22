import torch
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
            self.start = max(0,start)
            self.end = min(end,len(data_source))

            self.data_source = data_source

        def __iter__(self):

            return islice(iter(self.data_source),self.start, self.end)

        def __len__(self):
            return self.end - self.start
