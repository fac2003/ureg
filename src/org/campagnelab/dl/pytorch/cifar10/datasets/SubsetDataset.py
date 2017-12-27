from torchnet.dataset.dataset import Dataset


class SubsetDataset(Dataset):
    def __init__(self, source, indices, labels=None):
        super().__init__()
        self.source = source
        self.indices = indices
        self.labels = labels

    def __getitem__(self, idx):
        image, label = self.source[self.indices[idx]]
        if self.labels is not None:
            label = self.labels[idx]
        return (image, label)

    def __len__(self):
        return min(len(self.source), len(self.indices))
