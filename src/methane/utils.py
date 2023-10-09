import pathlib
from tqdm import tqdm
import torch.nn.init as init

import tifffile as tiff
import numpy as np
import pandas as pd

import torch.nn as nn


# Use a context manager (with statement) to open the TIFF file


def load_train(dir_name):
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
        return np.array(X_test), np.array(y_test), str(file_list)

    return np.array(X_test), np.array(y_test)


def weight_init(m):
    """
    Initializes a model's parameters.
    Credits to: https://gist.github.com/jeasinema

    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=0, std=1)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        try:
            init.normal_(m.bias.data)
        except AttributeError:
            pass
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
