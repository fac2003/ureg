import collections

import torch
import torchvision
from torch.utils.data.dataloader import default_collate, numpy_type_map, string_classes
from torchvision import transforms
from torchvision.transforms import Scale

from org.campagnelab.dl.pytorch.cifar10.Problem import Problem
from org.campagnelab.dl.pytorch.cifar10.Samplers import ProtectedSubsetRandomSampler


def stl10_collate(batch):
    """A custom collate function to handle the label of the unsupervised
         examples being None in STL10."""
    if torch.is_tensor(batch[0]):
        out = None
        return torch.stack(batch, 0, out=out)
    elif type(batch[0]).__module__ == 'numpy':
        elem = batch[0]
        if type(elem).__name__ == 'ndarray':
            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: stl10_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [stl10_collate(samples) for samples in transposed]
    elif batch[0] is None:
        # STL10 has None in place of label for the unlabeled split. We need to return None in this case instead of
        # failing, as would default_collate fail.
        return None

    raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))


class STL10Problem(Problem):
    """A problem that exposes the  Cifar10 dataset."""

    def name(self):
        return "STL10"

    def example_size(self):
        return (3, 96, 96)

    def __init__(self, mini_batch_size):
        super().__init__(mini_batch_size)
        from PIL import Image
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        self.trainset = torchvision.datasets.STL10(root='./data', split="train", download=True,
                                                   transform=self.transform_train)
        self.testset = torchvision.datasets.STL10(root='./data', split="test", download=False,
                                                  transform=self.transform_test)
        self.unsupset = torchvision.datasets.STL10(root='./data', split="unlabeled", download=False,
                                                   transform=self.transform_train)

    def train_loader(self):
        """Returns the torch dataloader over the training set. """

        mini_batch_size = self.mini_batch_size()

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=mini_batch_size, shuffle=False,
                                                  collate_fn=stl10_collate,
                                                  num_workers=2, drop_last=True)
        return trainloader

    def train_loader_subset(self, start, end):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        mini_batch_size = self.mini_batch_size()

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=mini_batch_size, shuffle=False,
                                                  sampler=ProtectedSubsetRandomSampler(
                                                      range(start, end)),
                                                  collate_fn=stl10_collate,
                                                  num_workers=2, drop_last=True)
        return trainloader

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self.testset,
                                           collate_fn=stl10_collate,
                                           batch_size=mini_batch_size, shuffle=False, num_workers=2, drop_last=True)

    def reg_loader(self):
        mini_batch_size = self.mini_batch_size()

        return torch.utils.data.DataLoader(self.unsupset, batch_size=mini_batch_size, shuffle=True,
                                           collate_fn=stl10_collate,
                                           num_workers=2, drop_last=True)

    def reg_loader_subset(self, start, end):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        # transform the unsupervised set the same way as the training set:

        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self.unsupset, batch_size=mini_batch_size, shuffle=False,
                                           sampler=ProtectedSubsetRandomSampler(range(start,
                                                                                      end)),
                                           collate_fn=stl10_collate,
                                           num_workers=2, drop_last=True)

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()
