"""
* 该脚本用于展示模型的结构和参数
*
* File: show_model.py
* Author: Fan Kai
* Soochow University
* Created: 2024-05-31 10:56:09
* ----------------------------
* Modified: 2024-06-14 21:02:16
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath("/workspace/src"))

# 模型目录
MODEL_DIR = r"/workspace/checkpoints/NN/V6_400320_mlp_3-200-200-200-1_relu_BS2_LR0.001_EP1000_2024-06-14T20-46-28"

# model
model_torch = torch.load(os.path.join(MODEL_DIR, "final_model.pth"))
model_torch.to("cpu")

with open(rf"{MODEL_DIR}/model_params.txt", "w") as f:
    for param_tensor in model_torch.state_dict():
        f.write(f"{param_tensor}\n")
        np_array = model_torch.state_dict()[param_tensor].detach().cpu().numpy()
        np.savetxt(f, np_array)


with open(rf"{MODEL_DIR}/model_architecture.txt", "w") as f:
    print(model_torch, file=f)
