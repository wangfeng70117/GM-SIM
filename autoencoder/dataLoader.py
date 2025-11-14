import numpy as np
import torch
from torch.utils.data import Dataset

class Autoencoder_dataset(Dataset):
    def __init__(self, features):
        features = np.load(features)
        print(f'features.shape is {features.shape}')
        self.data = torch.tensor(features, dtype=torch.float32)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.data.shape[0]