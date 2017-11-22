import torch
import torchvision
from torchvision import transforms

from org.campagnelab.dl.pytorch.cifar10.Problem import Problem


class Cifar10Problem(Problem):
    """A problem that exposes the  Cifar10 dataset."""

    def __init__(self):
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    def train_loader(self):
        """Returns the torch dataloader over the training set. """

        mini_batch_size = self.mini_batch_size
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                transform=self.transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=mini_batch_size, shuffle=False, num_workers=2)
        return trainloader

    def test_loader(self):
        """Returns the torch dataloader over the test set. """
        mini_batch_size = self.mini_batch_size
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_test)
        return torch.utils.data.DataLoader(testset, batch_size=mini_batch_size, shuffle=False, num_workers=2)


    def reg_loader(self):
        """Returns the torch dataloader over the regularization set (unsupervised examples only). """
        # transform the unsupervised set the same way as the training set:

        mini_batch_size = self.mini_batch_size
        unsupset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=self.transform_train)
        return torch.utils.data.DataLoader(unsupset, batch_size=mini_batch_size, shuffle=True, num_workers=2)

    def loss_function(self):
        return torch.nn.CrossEntropyLoss();

