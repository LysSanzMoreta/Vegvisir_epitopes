"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data_array_blosum,data_array_int,data_array_onehot,data_array_blosum_norm,batch_mask):
        self.batch_data_blosum = data_array_blosum
        self.batch_data_int = data_array_int
        self.batch_data_onehot = data_array_onehot
        self.batch_data_blosum_norm = data_array_blosum_norm
        self.batch_mask = batch_mask
    def __getitem__(self, index):  # sets a[i]
        batch_data_blosum = self.batch_data_blosum[index]
        batch_data_int = self.batch_data_int[index]
        batch_data_onehot = self.batch_data_onehot[index]
        batch_data_blosum_norm = self.batch_data_blosum_norm[index]
        batch_mask = self.batch_mask[index]
        return {'batch_data_blosum': batch_data_blosum,
                'batch_data_int':batch_data_int,
                'batch_data_onehot':batch_data_onehot,
                'batch_data_blosum_norm':batch_data_blosum_norm,
                'batch_mask':batch_mask}
    def __len__(self):
        return len(self.batch_data_blosum)

