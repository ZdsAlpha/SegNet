from torch.utils.data import Dataset

class Selector(Dataset):
    def __init__(self,loader,ids):
        self.ids = ids
        self.loader = loader
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self,index):
        return self.loader.__getitem__(self.ids[index])