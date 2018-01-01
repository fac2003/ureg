import collections

import torch
import torchvision
from torch.utils.data.dataloader import default_collate, numpy_type_map, string_classes
from torchvision import transforms
from torchvision.transforms import Scale, Grayscale
from torchvision.transforms.functional import to_grayscale

from org.campagnelab.dl.pytorch.images.Problem import Problem
from org.campagnelab.dl.pytorch.images.Samplers import ProtectedSubsetRandomSampler


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

    def train_set(self):
        return self._trainset

    def unsup_set(self):
        return self._unsupset

    def test_set(self):
        return self._testset

    def __init__(self, mini_batch_size, num_workers=0):
        super().__init__(mini_batch_size)
        self.num_workers=num_workers
        from PIL import Image
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),

        ])

        self.transform_test = transforms.Compose([
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._trainset = torchvision.datasets.STL10(root='./data', split="train", download=True,
                                                    transform=self.transform_train)
        self._testset = torchvision.datasets.STL10(root='./data', split="test", download=False,
                                                   transform=self.transform_test)
        self._unsupset = torchvision.datasets.STL10(root='./data', split="unlabeled", download=False,
                                                    transform=self.transform_train)

    def train_loader(self):
        """Returns the torch dataloader over the training set. """

        mini_batch_size = self.mini_batch_size()

        trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=mini_batch_size, shuffle=False,
                                                  collate_fn=stl10_collate,
                                                  num_workers=self.num_workers)
        return trainloader

    def train_loader_subset(self, indices):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        mini_batch_size = self.mini_batch_size()

        trainloader = torch.utils.data.DataLoader(self._trainset, batch_size=mini_batch_size, shuffle=False,
                                                  sampler=ProtectedSubsetRandomSampler(
                                                      indices),
                                                  collate_fn=stl10_collate,
                                                  num_workers=self.num_workers)
        return trainloader

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self._testset,
                                           collate_fn=stl10_collate,
                                           batch_size=mini_batch_size, shuffle=False,
                                           num_workers=self.num_workers)

    def test_loader_subset(self, indices):
        """Returns the torch dataloader over the test set, limiting to the examples
        identified by the indices. """

        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self._testset,
                                           collate_fn=stl10_collate,
                                           sampler=ProtectedSubsetRandomSampler(indices),
                                           batch_size=mini_batch_size, shuffle=False,
                                           num_workers=self.num_workers)

    def reg_loader(self):
        mini_batch_size = self.mini_batch_size()

        return torch.utils.data.DataLoader(self._unsupset, batch_size=mini_batch_size, shuffle=True,
                                           collate_fn=stl10_collate,
                                           num_workers=self.num_workers)

    def reg_loader_subset(self, indices):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        # transform the unsupervised set the same way as the training set:

        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self._unsupset, batch_size=mini_batch_size, shuffle=False,
                                           sampler=ProtectedSubsetRandomSampler(indices),
                                           collate_fn=stl10_collate,
                                           num_workers=self.num_workers)

    def loader_for_dataset(self, dataset):
        mini_batch_size = self.mini_batch_size()

        return torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=False,
                                       sampler=ProtectedSubsetRandomSampler(range(0,len(dataset))),
                                       collate_fn=stl10_collate,
                                       num_workers=self.num_workers)

    def num_classes(self):
        return 10

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()
