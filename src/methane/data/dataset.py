from typing import Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(
        self, features, targets, transform=True, normalize=None, extra_feature=None
    ) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform
        self.extra_feature = extra_feature

    def __getitem__(self, index) -> Any:
        targets = self.targets[index]
        features = self.features[index]

        if self.transform:
            features = reshape_transform(features)

        if normalize is not None:
            features = normalize(features, normalize[0], normalize[1])

        if self.extra_feature is not None:
            extra_feature = self.extra_feature[index]
            if normalize is not None:
                extra_feature = normalize(extra_feature, normalize[0], normalize[1])
            if self.transform:
                extra_feature = reshape_transform(extra_feature)
            return [torch.cat([features, extra_feature], dim=0), targets]

        return [features, targets]

    def __len__(self):
        return len(self.targets)


def reshape_transform(x):
    x = x.view(1, x.shape[0], x.shape[1])  # Reshape to (1, H, W)
    return x


def normalize(x, mean, std):
    x = transforms.Normalize(x, mean, std)
    return
