from typing import Any

from torch.utils.data import Dataset


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

        return [features, targets]

    def __len__(self):
        """
        Get the total number of items in the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.targets)


def reshape_transform(x):
    x = x.view(1, x.shape[0], x.shape[1])  # Reshape to (1, 1, H, W)
    return x
