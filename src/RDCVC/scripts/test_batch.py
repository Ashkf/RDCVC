"""
实际使用时需要将训练好的模型上在输入数据上运行，这里以测试集的数据为例

torch.no_grad()
    停止 autograd 模块的工作，不计算和储存梯度，一般在用训练好的模型跑测试集时使用，
    因为测试集时不需要计算梯度更不会更新梯度。使用后可以加速计算时间，节约 gpu 的显存
"""

import os
import pickle
import sys

import pandas as pd

sys.path.append(".")
sys.path.append("/workspace/src")

import torch


@torch.no_grad()
def main():
    CKP_DIR = r"checkpoints/cvcnet-mtl-mlp_18_1_2_3_64-64-64_64-64-64_BS16_LR0.01_EP10000_2024-02-26T15-33-04"
    TEST_DATA_PATH = r"/workspace/data/test/tmp_rdc_data_cleaned_eval.csv"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载训练好的模型
    nnmodel_path = os.path.join(CKP_DIR, "final_model.pth")
    nnmodel = torch.load(nnmodel_path).to(DEVICE)
    nnmodel.eval()

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

    data_raw = torch.tensor(_df[data_key].copy().values, dtype=torch.float32).to(DEVICE)
    target = torch.tensor(_df[target_key].copy().values, dtype=torch.float32).to(DEVICE)

    # ------------------------ 准备归一化器 ------------------------ #
    with open(os.path.join(CKP_DIR, "x_normalizer.pkl"), "rb") as f:
        x_scaler = pickle.load(f)
    with open(os.path.join(CKP_DIR, "y_normalizer.pkl"), "rb") as f:
        y_scaler = pickle.load(f)

    # -------------------------- 预测 -------------------------- #
    pred_raw = nnmodel(x_scaler.transform(data_raw))
    pred = y_scaler.inverse_transform(torch.cat(pred_raw, dim=1))

    # 将 target 和 preds 水平连接 (numpy)
    combined = torch.cat([target, pred], dim=1).cpu().numpy()

    # 将数据框保存为 CSV 文件 (Dataframe)
    pd.DataFrame(
        combined, columns=target_key + ["preds_" + k for k in target_key]
    ).to_csv("/workspace/output.csv", index=False)


if __name__ == "__main__":
    main()
