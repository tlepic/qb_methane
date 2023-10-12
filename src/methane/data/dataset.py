from typing import Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    """
    A custom PyTorch dataset for working with image data, designed to be used with the DataLoader class.

    Args:
        features (list): A list of image features or paths.
        targets (list): A list of corresponding target labels.
        transform (bool, optional): If True, applies a transformation to the image features during retrieval.
                                    Default is True.

    Methods:
        __getitem__(self, index): Retrieves a single item (feature, target) from the dataset.
        __len__(self): Returns the total number of items in the dataset.

    Example:
        >>> dataset = ImageDataset(features, targets, transform=True)
        >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    """
    def __init__(self, features, targets, transform=True) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform
        self.extra_feature = extra_feature

    def __getitem__(self, index) -> Any:
        """
        A custom PyTorch dataset for working with image data, designed to be used with the DataLoader class.

        Args:
            features (list): A list of image features or paths.
            targets (list): A list of corresponding target labels.
            transform (bool, optional): If True, applies a transformation to the image features during retrieval.
                                        Default is True.

        Methods:
            __getitem__(self, index): Retrieves a single item (feature, target) from the dataset.
            __len__(self): Returns the total number of items in the dataset.

        Example:
            >>> dataset = ImageDataset(features, targets, transform=True)
            >>> dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        """
        targets = self.targets[index]
        features = self.features[index]

        if self.transform:
            features = reshape_transform(features)

        if self.extra_feature is not None:
            extra_feature = self.extra_feature[index]
            if self.transform:
                extra_feature = reshape_transform(extra_feature)
            return [torch.cat([features, extra_feature], dim=0), targets]

        return [features, targets]

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.targets)


def reshape_transform(x):
    x = x.view(1, x.shape[0], x.shape[1])  # Reshape to (1, H, W)
    return x
