import os
import pathlib
import numpy as np
import pandas as pd
import tifffile as tiff
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm

def load_train(dir_name):
    """
    Load training data from TIFF images and corresponding metadata.

    Args:
        dir_name (str): The directory containing the training data.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of training features (image data).
            - np.ndarray: An array of training labels (0.0 for no plume, 1.0 for plume).

    Note:
        This function reads TIFF images and their corresponding labels from the specified directory,
        converting the labels to numeric values (0.0 or 1.0) based on the 'plume' column in the metadata.

    Example:
        >>> X_train, y_train = load_train("data_directory")
    """
    data_dir = pathlib.Path(dir_name) / "train_data"

    X_train = []
    y_train = []

    df_train = pd.read_csv(data_dir / "metadata.csv")

    for sample, plume in tqdm(df_train[["path", "plume"]].values):
        sample_path = data_dir / sample
        with tiff.TiffFile(str(sample_path) + ".tif") as tif:
            _feature = tif.asarray().astype(np.float64)

        X_train.append(_feature)
        if plume == "yes":
            y_train.append(1.0)
        else:
            y_train.append(0.0)

    print(
        "Training set specifications\n"
        "---------------------------\n"
        f"{len(X_train)} unique samples\n"
    )

    return np.array(X_train), np.array(y_train)


def load_test(dir_name, return_path=False):
    """
    Load test data from TIFF images.

    Args:
        dir_name (str): The directory containing the test data.
        return_path (bool, optional): If True, returns a list of file paths along with the test data.
                                      Default is False.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: An array of test features (image data).
            - np.ndarray: An array of dummy test labels (always 0.0, used for compatibility).
            - list: A list of file names if 'return_path' is True; otherwise, only the features and dummy labels.

    Note:
        This function reads TIFF images from the specified directory and returns them as features.
        It also provides the option to return a list of file names if 'return_path' is set to True.

    Example:
        >>> X_test, y_test = load_test("test_data_directory")
        # or
        >>> X_test, y_test, file_paths = load_test("test_data_directory", return_path=True)
    """
    data_dir = pathlib.Path(dir_name) / "test_data" / "images"

    file_list = [file for file in data_dir.iterdir() if file.is_file()]

    X_test = []
    y_test = []  # Warning this is used only for compatibility

    for sample in tqdm(file_list):
        with tiff.TiffFile(sample) as tif:
            _feature = tif.asarray().astype(np.float64)

        X_test.append(_feature)
        y_test.append(0.0)

    print(
        "Testing set specifications\n"
        "---------------------------\n"
        f"{len(X_test)} unique samples\n"
    )
    if return_path:
        return np.array(X_test), np.array(y_test), os.listdir(data_dir)

    return np.array(X_test), np.array(y_test)
