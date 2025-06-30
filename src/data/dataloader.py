import h5py
import random
import torch
from torch.utils.data import Dataset

class DRRDataset(Dataset):
    def __init__(
        self, hdf5_file, transform_view=None, transform_corr=None, data_type="train"
    ):
        self.hdf5_file = hdf5_file
        self.transform_view = transform_view
        self.transform_corr = transform_corr
        self.data_type = data_type
        with h5py.File(hdf5_file, "r") as hf:
            self.indices = list(hf[data_type].keys())

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, "r") as hf:
            name = self.indices[idx]
            group = hf[f"{self.data_type}/{name}"]
            view1 = torch.from_numpy(group["view1"][()])
            view2 = torch.from_numpy(group["view2"][()])
            corr = torch.from_numpy(group["corr1_2"][()])

        if self.transform_view:
            view1 = self.transform_view(view1).squeeze()
            view2 = self.transform_view(view2).squeeze()
        if self.transform_corr:
            if isinstance(self.transform_corr, tuple):
                augmentations, probabilities = self.transform_corr
                for transform, probability in zip(augmentations, probabilities):
                    if random.random() < probability:
                        corr, view1, view2 = transform(corr, view1, view2)
            else:
                corr, view1, view2 = self.transform_corr(corr, view1, view2)

        view1 = view1.unsqueeze(0) if view1.dim() == 2 else view1
        view2 = view2.unsqueeze(0) if view2.dim() == 2 else view2
        views = torch.stack([view1, view2], dim=0)

        return corr, views