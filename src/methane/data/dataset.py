from typing import Any
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, features, targets, transform=True, extra_feature=None) -> None:
        """
        Custom dataset class for handling image data.

        Args:
            features (torch.Tensor): Tensor containing the image features.
            targets (torch.Tensor): Tensor containing the corresponding targets.
            transform (bool, optional): Flag indicating whether to apply transformations to the features. Default is True.
            extra_feature (torch.Tensor, optional): Tensor containing additional features. Default is None.
        """
    def __init__(self, features, targets, extra_feature, transform=True) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.transform = transform
        self.extra_feature = extra_feature

    def __getitem__(self, index) -> Any:
        """
        Retrieves an item from the dataset.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            list: A list containing the features and targets.
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
        Returns the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.targets)


def reshape_transform(x):
    """
    Reshapes the input tensor.

    Args:
        x (torch.Tensor): Input tensor to reshape.

    Returns:
        torch.Tensor: Reshaped tensor.
    """
    x = x.view(1, x.shape[0], x.shape[1])  # Reshape to (1, H, W)
    return x
