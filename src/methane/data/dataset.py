from typing import Any
import logging
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, features, targets, transform=True, extra_feature=None) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform
        self.extra_feature = extra_feature

    def __getitem__(self, index) -> Any:
        targets = self.targets[index]
        features = self.features[index]

        if self.transform:
            features = self.transform(features)
        features = reshape_transform(features)

        if self.extra_feature is not None:
            extra_feature = self.extra_feature[index]
            if self.transform:
                extra_feature = reshape_transform(extra_feature)
            return [torch.cat([features, extra_feature], dim=0), targets]

        return [features, targets]


    def __len__(self):
        return len(self.targets)

class CheckedImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # If there's a transform, apply it
        if self.transform:
            image = self.transform(image)
        return image.clone().detach(), label.clone().detach()

def reshape_transform(x):
    x = x.view(1, x.shape[0], x.shape[1])  # Reshape to (1, H, W)
    return x
