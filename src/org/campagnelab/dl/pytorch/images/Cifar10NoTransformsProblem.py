import torch
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from org.campagnelab.dl.pytorch.images.Problem import Problem
from org.campagnelab.dl.pytorch.images.Samplers import ProtectedSubsetRandomSampler


class Cifar10_NT64Problem(Problem):
    """A problem that exposes the  Cifar10 dataset, no transformations applied, but images resized to 64x64."""

    def name(self):
        return "CIFAR10_NT64"

    def example_size(self):
        return (3, 64,64)

    def train_set(self):
        return self.trainset

    def unsup_set(self):
        return self.unsupset

    def test_set(self):
        return self.testset

    def __init__(self, mini_batch_size):
        super().__init__(mini_batch_size)
        self.transform_train = transforms.Compose([
            transforms.Resize((64,64)),
            #transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])

        self.transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)),
        ])
        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                     transform=self.transform_train)
        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                    transform=self.transform_test)
        self.unsupset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                                     transform=self.transform_train)

    def num_classes(self):
        return 10

    def loader_for_dataset(self, dataset):
        mini_batch_size = self.mini_batch_size()

        return torch.utils.data.DataLoader(dataset, batch_size=mini_batch_size, shuffle=False,
                                           sampler=ProtectedSubsetRandomSampler(range(0, len(dataset)))
                                           )

    def train_loader(self):
        """Returns the torch dataloader over the training set. """

        mini_batch_size = self.mini_batch_size()

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=mini_batch_size, shuffle=False,
                                                  num_workers=2)
        return trainloader

    def train_loader_subset(self, indices):
        """Returns the torch dataloader over the training set, shuffled,
        but limited to the example range start-end."""
        mini_batch_size = self.mini_batch_size()

        trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=mini_batch_size, shuffle=False,
                                                  sampler=ProtectedSubsetRandomSampler(indices),
                                                  num_workers=2)
        return trainloader

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self.testset, batch_size=mini_batch_size, shuffle=False, num_workers=2)

    def test_loader_subset(self, indices):
        """Returns the torch dataloader over the test set. """
        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self.testset,
                                           sampler=ProtectedSubsetRandomSampler(indices),
                                           batch_size=mini_batch_size, shuffle=False, num_workers=2)

    def reg_loader(self):
        mini_batch_size = self.mini_batch_size()

        return torch.utils.data.DataLoader(self.unsupset, batch_size=mini_batch_size, shuffle=False,
                                           num_workers=2)

    def reg_loader_subset(self, indices):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        # transform the unsupervised set the same way as the training set:

        mini_batch_size = self.mini_batch_size()
        return torch.utils.data.DataLoader(self.unsupset, batch_size=mini_batch_size, shuffle=False,
                                           sampler=ProtectedSubsetRandomSampler(indices),
                                           num_workers=2)

    def loss_function(self):
        return torch.nn.CrossEntropyLoss()
