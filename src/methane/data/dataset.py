from typing import Any

from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, features, targets, transform=True) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index) -> Any:
        targets = self.targets[index]
        features = self.features[index]

        if self.transform:
            features = reshape_transform(features)

        return [features, targets]

    def __len__(self):
        return len(self.targets)


def reshape_transform(x):
    x = x.view(1, x.shape[0], x.shape[1])  # Reshape to (1, 1, H, W)
    return x
