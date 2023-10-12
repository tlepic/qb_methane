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
    data_dir = pathlib.Path(dir_name) / "train_data"

    X_train = []
    y_train = []
    X_extra_feature = []

    df_train = pd.read_csv(data_dir / "metadata.csv")

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


def load_test(dir_name, return_path=False):
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


def encode_positions(coord_x, coord_y, matrix_size=64):
    matrix = np.empty((matrix_size, matrix_size), dtype=object)

    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = (-i, j)

    # Adding (10, 10) to each element of the matrix
    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = (matrix[i, j][0] + coord_x, matrix[i, j][1] - coord_y)

    for i in range(matrix_size):
        for j in range(matrix_size):
            matrix[i, j] = np.sqrt(matrix[i, j][0] ** 2 + matrix[i, j][1] ** 2)

    r = np.max(matrix.any())
    matrix = matrix / r
    return np.array(matrix.astype(np.float64))
