"""
* for process datasets.
*
* File: datasets.py
* Author: Fan Kai
* Soochow University
* Created: 2023-10-07 02:26:02
* ----------------------------
* Modified: 2023-11-19 11:20:02
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""


import pandas as pd
from sklearn.model_selection import train_test_split


def load_and_split_data(data_file, test_size=0.1, random_state=42):
    """
    Loads the data from a CSV file, selects the relevant features,
    and splits the dataset into training and testing sets.

    Args:
        data_file (str): The path to the CSV file containing the data.
        test_size (float, optional): The proportion of the dataset to
            include in the testing set. Defaults to 0.1.
        random_state (int, optional): Controls the shuffling applied
            to the data before splitting. Defaults to 42.

    Returns:
        x_train (ndarray): The feature values for the training set.
        x_test (ndarray): The feature values for the testing set.
        y_train (ndarray): The target values for the training set.
        y_test (ndarray): The target values for the testing set.

    """
    _df_raw = pd.read_csv(data_file)

    decision_features = [
        "MAU_FREQ",
        "AHU_FREQ",
        "EF_FREQ",
        "RM1_SUPP_DMPR_0",
        "RM2_SUPP_DMPR_0",
        "RM6_SUPP_DMPR_0",
        "RM6_SUPP_DMPR_1",
        "RM3_SUPP_DMPR_0",
        "RM4_SUPP_DMPR_0",
        "RM5_SUPP_DMPR_0",
        "RM2_RET_DMPR_0",
        "RM6_RET_DMPR_0",
        "RM3_RET_DMPR_0",
        "RM4_RET_DMPR_0",
        "RM5_EXH_DMPR_1",
        "RM3_EXH_DMPR_0",
        "RM4_EXH_DMPR_0",
        "RM5_EXH_DMPR_0",
    ]
    controlled_features = [
        "TOT_FRSH_VOL",
        "TOT_SUPP_VOL",
        "TOT_RET_VOL",
        "TOT_EXH_VOL",
        "RM1_PRES",
        "RM2_PRES",
        "RM3_PRES",
        "RM4_PRES",
        "RM5_PRES",
        "RM6_PRES",
    ]

    _df_data = _df_raw[decision_features]
    _df_label = _df_raw[controlled_features]

    x_train, x_test, y_train, y_test = train_test_split(
        _df_data.values,
        _df_label.values,
        test_size=test_size,
        shuffle=True,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test
