"""
* 神经网络预测脚本，用于测试单次预测
*
* File: test.py
* Author: Fan Kai
* Soochow University
* Created: 2024-03-09 23:01:26
* ----------------------------
* Modified: 2024-03-09 23:06:49
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import pickle

import torch

nndir = r""
model_path = os.path.join(nndir, "final_model.pth")
input_data = {"前阀片组开度": 1, "后阀片开度": 2, "风阀读取压差（Pa）": 3}

nnmodel = torch.load(model_path)  # 加载神经网络模型
nnmodel.eval()
print(f"输入：{input_data}")
data = torch.Tensor(
    [
        [
            input_data["前阀片组开度"],
            input_data["后阀片开度"],
            input_data["风阀读取压差（Pa）"],
        ]
    ]
)
data.requires_grad_(False)

# ============= 归一化 =============
scaler_path = os.path.join(nndir, "scalers_dict.pkl")  # 归一化器路径
with open(scaler_path, "rb") as f:
    scalers = pickle.load(f)
items_scalered = scalers["scaler_in"].transform(data)
# ============= 预测 =============
pred = nnmodel(torch.Tensor(items_scalered))
# ============= 反归一化 =============
pred = scalers["scaler_out"].inverse_transform(pred.detach().numpy())
print(f"预测风量（m3/h）：{pred[0][0]}")
