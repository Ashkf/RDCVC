"""
* 神经网络推理脚本，用于实现单次预测
*
* File: nn_infer.py
* Author: Fan Kai
* Soochow University
* Created: 2024-07-15 09:10:16
* ----------------------------
* Modified: 2024-07-15 13:01:27
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import pickle
import sys

import pandas as pd
import torch

sys.path.append("/workspace/src")

CKP_DIR = (
    r"checkpoints/NN/cvcnet-mtl-mlp_18_1_2_3_64-64-64_64-64-64"
    r"_BS16_LR0.01_EP10000_2024-07-15T12-29-54"
)
TEST_DATA_PATH = r"/workspace/data/test/MNSGA31_20240715T001557.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main():
    nnmodel = load_model(model_path=os.path.join(CKP_DIR, "final_model.pth"))
    data_key, data_raw = load_data()
    x_scaler, y_scaler = load_scaler()

    # -------------------------- 预测 -------------------------- #
    pred_raw = nnmodel(x_scaler.transform(data_raw))
    pred = y_scaler.inverse_transform(torch.cat(pred_raw, dim=1))

    target_key = [
        "TOT_FRSH_VOL",
        "TOT_SUPP_VOL",
        "TOT_EXH_VOL",
        "TOT_RET_VOL",
        "RM1_PRES",  # a
        "RM2_PRES",  # b
        "RM3_PRES",  # d
        "RM4_PRES",  # e
        "RM5_PRES",  # f
        "RM6_PRES",  # c
    ]
    print(f"预测结果：{pd.DataFrame(pred.cpu(), columns=target_key)}")


def load_scaler() -> tuple:
    with open(os.path.join(CKP_DIR, "x_normalizer.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(CKP_DIR, "y_normalizer.pkl"), "rb") as f:
        y_scaler = pickle.load(f)

    x_scaler.device = DEVICE
    y_scaler.device = DEVICE
    return x_scaler, y_scaler


def load_model(model_path: str):
    nnmodel = torch.load(model_path)  # 加载神经网络模型
    nnmodel.eval()
    return nnmodel


def load_data() -> tuple[list[str], torch.Tensor]:
    data_key = [
        "MAU_FREQ",
        "AHU_FREQ",
        "EF_FREQ",
        "RM1_SUPP_DMPR_0",
        "RM2_SUPP_DMPR_0",
        "RM3_SUPP_DMPR_0",
        "RM4_SUPP_DMPR_0",
        "RM5_SUPP_DMPR_0",
        "RM6_SUPP_DMPR_0",
        "RM6_SUPP_DMPR_1",
        "RM2_RET_DMPR_0",
        "RM3_RET_DMPR_0",
        "RM4_RET_DMPR_0",
        "RM6_RET_DMPR_0",
        "RM3_EXH_DMPR_0",
        "RM4_EXH_DMPR_0",
        "RM5_EXH_DMPR_0",
        "RM5_EXH_DMPR_1",
    ]
    _df = pd.read_csv(TEST_DATA_PATH)

    data = torch.tensor(_df[data_key].copy().values, dtype=torch.float32).to(DEVICE)
    return data_key, data


if __name__ == "__main__":
    main()
