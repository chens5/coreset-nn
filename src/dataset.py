from torch.utils.data import Dataset, DataLoader

class CoresetDataset(Dataset):
    def __init__(self, pointset, coreset):
        self.pointset = pointset
        self.coreset_mask = coreset
    
    def __len__(self):
        return len(self.sources)

    def __getitem__(self, idx):
        return self.pointset[idx], self.coreset_mask[idx]

