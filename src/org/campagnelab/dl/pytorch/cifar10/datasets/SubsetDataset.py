from torchnet.dataset.dataset import Dataset


class SubsetDataset(Dataset):
    def __init__(self, source, indices):
        super().__init__()
        self.source=source
        self.indices=indices

    def __getitem__(self, idx):

        return self.source[self.indices[idx]]

    def __len__(self):

        return min(len(self.source),len(self.indices))