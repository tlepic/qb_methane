from typing import Any
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, features, targets, transform=True, extra_feature=None) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform
        self.extra_feature = extra_feature

    def __getitem__(self, index) -> Any:
        targets = self.targets[index]
        features = torch.tensor(self.features[index])

        if self.transform:
            features = reshape_transform(features)

        if self.extra_feature is not None:
            extra_feature = self.extra_feature[index]
            if self.transform:
                extra_feature = reshape_transform(extra_feature)
            return [torch.cat([features, extra_feature], dim=0), targets]

        return [features, targets]

    def __len__(self):
        return len(self.targets)


def reshape_transform(x):
    return x.view(1, x.shape[0], x.shape[1])
