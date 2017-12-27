from torchnet.dataset.dataset import Dataset


class SubsetDataset(Dataset):
    def __init__(self, source, indices, get_label=None):
        super().__init__()
        self.source = source
        self.indices = indices
        self.get_label = get_label

    def __getitem__(self, idx):
        image, label = self.source[self.indices[idx]]
        if self.get_label is not None:
            label = self.get_label(idx)
        return (image, label)

    def __len__(self):
        return min(len(self.source), len(self.indices))
