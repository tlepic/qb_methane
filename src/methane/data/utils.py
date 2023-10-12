import os
import pathlib

import numpy as np
import pandas as pd
import tifffile as tiff
import torch.nn as nn
import torch.nn.init as init
from tqdm import tqdm

# Use a context manager (with statement) to open the TIFF file


def load_train(dir_name, extra_feature=False):
    """
    Load training data from a specified directory and create a dataset for machine learning.

    Parameters:
        dir_name (str): The directory containing the training data.
        extra_feature (bool, optional): Whether to include extra features based on coordinate information.
            If True, additional features are created by encoding positions of coordinates and
            multiplying them with the image data. Default is False.

    Returns:
        If extra_feature is True:
            - X_train (ndarray): An array containing the training samples (image data).
            - y_train (ndarray): An array containing the corresponding binary labels (1.0 for 'plume' and 0.0 for 'no plume').
            - X_extra_feature (ndarray): An array containing extra features if extra_feature is True.
        If extra_feature is False:
            - X_train (ndarray): An array containing the training samples (image data).
            - y_train (ndarray): An array containing the corresponding binary labels (1.0 for 'plume' and 0.0 for 'no plume').
    """
    data_dir = pathlib.Path(dir_name) / "train_data"

    X_train = []
    y_train = []
    X_extra_feature = []

    df_train = pd.read_csv(data_dir / "metadata.csv")
    df_train = df_train.drop_duplicates(subset=["date", "id_coord"])

    for sample, plume, coord_x, coord_y in tqdm(
        df_train[["path", "plume", "coord_x", "coord_y"]].values
    ):
        sample_path = data_dir / sample
        with tiff.TiffFile(str(sample_path) + ".tif") as tif:
            _feature = tif.asarray().astype(np.float64)

        _extra_feature = np.multiply(encode_positions(coord_x, coord_y), _feature)
        if extra_feature:
            X_extra_feature.append(_extra_feature)
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

    if extra_feature:
        return np.array(X_train), np.array(y_train), np.array(X_extra_feature)

    return np.array(X_train), np.array(y_train)


def load_test(dir_name, extra_feature=False, return_path=False):
    """
    Loads the test data for methane detection.

    Args:
        dir_name (str): Directory name where the test data is located.
        extra_feature (bool, optional): Flag indicating whether to include extra features. Default is False.
        return_path (bool, optional): Flag indicating whether to return the file paths instead of file names. Default is False.

    Returns:
        tuple or numpy.ndarray: If return_path is True and extra_feature is True, returns a tuple containing X_test (features), y_test (targets), X_extra_feature (extra features), and file_list (file paths).
        If return_path is True and extra_feature is False, returns a tuple containing X_test, y_test, and file_list.
        If return_path is False and extra_feature is True, returns a tuple containing X_test, y_test, and X_extra_feature.
        If return_path is False and extra_feature is False, returns a tuple containing X_test and y_test.
    """
    data_dir = pathlib.Path(dir_name) / "test_data"

    df_test = pd.read_csv(data_dir / "metadata.csv")
    df_test = df_test.drop_duplicates(subset=["date", "id_coord"])
    df_test["path"] = df_test.apply(
        lambda row: f"{data_dir}/images/{row['date']}_methane_mixing_ratio_{row['id_coord']}.tif",
        axis=1,
    )
    df_test["result_path"] = df_test.apply(
        lambda row: f"{row['date']}_methane_mixing_ratio_{row['id_coord']}.tif",
        axis=1,
    )

    file_list = os.listdir(data_dir / "images")

    X_test = []
    X_extra_feature = []
    y_test = []  # Warning this is used only for compatibility

    for sample, coord_x, coord_y in tqdm(
        df_test[["path", "coord_x", "coord_y"]].values
    ):
        with tiff.TiffFile(sample) as tif:
            _feature = tif.asarray().astype(np.float64)
        _extra_feature = np.multiply(encode_positions(coord_x, coord_y), _feature)

        X_test.append(_feature)
        y_test.append(0.0)
        if extra_feature:
            X_extra_feature.append(_extra_feature)

    print(
        "Testing set specifications\n"
        "---------------------------\n"
        f"{len(X_test)} unique samples\n"
    )

    if return_path:
        file_list = df_test["result_path"]
        if extra_feature:
            return (
                np.array(X_test),
                np.array(y_test),
                np.array(X_extra_feature),
                file_list,
            )
        return np.array(X_test), np.array(y_test), file_list

    if extra_feature:
        return np.array(X_test), np.array(y_test), np.array(X_extra_feature)
    return np.array(X_test), np.array(y_test)


def encode_positions(coord_x, coord_y, matrix_size=64):
    """
    Encodes the positions based on the given coordinates.

    Args:
        coord_x (float): X-coordinate.
        coord_y (float): Y-coordinate.
        matrix_size (int, optional): Size of the matrix. Default is 64.

    Returns:
        numpy.ndarray: Encoded positions matrix.
    """
    matrix = np.empty((matrix_size, matrix_size), dtype=object)

    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = (-i, j)

    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = (matrix[i, j][0] + coord_x, matrix[i, j][1] - coord_y)

    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = np.sqrt(matrix[i, j][0] ** 2 + matrix[i, j][1] ** 2)

    r = np.max(matrix.any())
    matrix = matrix / r
    return np.array(matrix.astype(np.float64))
