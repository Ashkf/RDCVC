"""
* 该脚本用于模型的发布。
* 包括模型的加载、测试、转换为 ONNX 模型、测试 ONNX 模型、
* 准备使用案例、保存使用案例、保存标准化器参数等
*
* File: release.py
* Author: Fan Kai
* Soochow University
* Created: 2023-10-09 11:00:54
* ----------------------------
* Modified: 2024-06-08 16:38:30
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import pickle
import sys

import numpy as np
import onnx
import onnxruntime
import torch
import yaml

sys.path.append("/workspace/src")
# -------------------------------------------------------- #
#                        load model                        #
# -------------------------------------------------------- #
print("> start loading model...", end="")
# 从 checkpoint 或 final model 中加载模型

# 1. 从 checkpoint 中加载模型（by state_dict）
# model.load_state_dict(
#     torch.load(
#         r"D:\OneDrive\01 WORK\# DampersClusterControl\03 IoT damper"
#         r" fitting\checkpoints\mlp-l12-dpTv_BS36_LR0.001_EP500_2023-08-20T21-10-53\ckps\ckp_E0490-B0000.pth"
#     )["model_state_dict"]
# )


# 2. 从 final model 中加载模型（by 完整模型）
ckp_dir = r"/workspace/src/IoTdamper/ckps/dapn12_BS8_LR0.001_EP10000_2023-10-09T10-36-49"


path_release = os.path.join(ckp_dir, "release")
path_onnx = os.path.join(path_release, "final_model.onnx")
os.makedirs(path_release, exist_ok=True)


# --------------- load model by state_dict --------------- #
from src.RDCVC.models.prediciton.CVCNet.models.IoTDamper_mlp import DAPN12

model_torch = DAPN12()
model_torch.load_state_dict(torch.load(os.path.join(ckp_dir, "ckps/ckp_E0174-B0000.pth"))["model_state_dict"])
# --------------- load model by full model --------------- #
# model_torch = torch.load(os.path.join(ckp_dir, "final_model.pth"))
print("done.")
# -------------------------------------------------------- #
#                        test model                        #
# -------------------------------------------------------- #
print("> start testing model...")

# ------------------- test torch model ------------------- #
print(">> test torch model")
_dummy_torch_in = torch.tensor([1, 2, 3], dtype=torch.float32)
_dummy_torch_out = model_torch(_dummy_torch_in)
print("torch native model inference test:\n" + "-" * 20)
print(f"torch_in: {_dummy_torch_in}, torch_out: {_dummy_torch_out}")


# ------------------- convert to onnx ------------------- #
def convert_onnx(model, onnx_path):
    """将 torch 模型转换为 ONNX 模型

    Args:
        model (torch.nn.Module): 待转换模型
        onnx_path (str): 目的 ONNX 文件名
    """
    model.eval()  # 设置为评估模式
    batch_size = 1  # 批处理大小
    input_shape = 3  # 输入数据长度
    dummy_input = torch.randn(batch_size, input_shape, requires_grad=False).to(next(model.parameters()).device)
    export_onnx_file = onnx_path  # 目的 ONNX 文件名

    # # ------------ remove nn.DataParallel wrapper ------------ #
    # state_dict = model.state_dict()
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     if k[:7] == "module.":
    #         name = k[7:]  # remove "module."
    #         new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # model.eval()

    torch.onnx.export(
        model.module if model.__class__.__name__ == "DataParallel" else model,  # 待转换模型
        dummy_input,  # 输入张量
        export_onnx_file,  # 目的 ONNX 文件名
        opset_version=10,  # ONNX 版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=["modelInput"],  # 输入名
        output_names=["modelOutput"],  # 输出名
        verbose=False,
    )


# 测试模型可用性
def check_onnx(model):
    """检测 ONNX 模型是否可用

    Args:
        model (onnx.ModelProto): ONNX 模型
    """
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print(f"The model is invalid: {e}")
    else:
        print("ONNX is valid.")


print(">> convert to onnx...")
convert_onnx(model_torch, path_onnx)
model_onnx = onnx.load(path_onnx)
check_onnx(model_onnx)

# ----------------------- inference test ----------------------- #
print(">> test inference...")

# onnxruntime.InferenceSession 用于获取一个 ONNX Runtime 推理器
ort_session = onnxruntime.InferenceSession(path_onnx)
# 构建字典的输入数据，字典的 key 需要与我们构建 onnx 模型时的 input_names 相同
ort_inputs = {"modelInput": np.array([[1, 2, 3]]).astype(np.float32)}
# run 进行模型的推理
ort_output = ort_session.run(None, ort_inputs)[0]


print("onnx model inference test:\n" + "-" * 20)
print(f"onnx_in: {ort_inputs}, onnx_out: {ort_output}")

if np.allclose(_dummy_torch_out.detach().cpu().numpy(), ort_output, rtol=1e-03, atol=1e-05):
    print("Torch model and onnx model outputs are consistent.")
print("> test done.")
# -------------------------------------------------------- #
#                          release                         #
# -------------------------------------------------------- #
print("> start release...")

# -------------------- prepare usecase ------------------- #
print(">> prepare use case")
# scaler_x
scaler_path = os.path.join(ckp_dir, "x_normalizer.pkl")
with open(scaler_path, "rb") as f:
    scaler_x = pickle.load(f)

# scaler_y
from sklearn.preprocessing import StandardScaler

scaler_y = StandardScaler()
scaler_y.mean_ = np.array([0])
scaler_y.scale_ = np.array([1])
scaler_y.var_ = np.array([1])


def save_usecase(usecase_input: list[int], usecase_path: str):
    usecase_input = np.array([usecase_input]).astype(np.float32)
    usecase_input.astype(np.float32)
    usecase_onnx_input = {
        "modelInput": scaler_x.transform(usecase_input)
        if isinstance(usecase_input, np.ndarray)
        else scaler_x.transform(usecase_input.numpy())
    }
    usecase_onnx_output = ort_session.run(None, usecase_onnx_input)
    usecase_output = scaler_y.inverse_transform(usecase_onnx_output[0].reshape(-1, 1))

    with open(usecase_path, "a") as f:
        f.write("\n")
        f.write("use case\n" + "-" * 20 + "\n")
        f.write(f"输入数据：{usecase_input}\n")
        f.write(f"ONNX 输入：{usecase_onnx_input}\n")
        f.write(f"ONNX 输出：{usecase_onnx_output}\n")
        f.write(f"输出数据：{usecase_output}\n")


# rls_input = np.array([[11, 33, 88]]).astype(np.float32)

# rls_onnx_input = {"modelInput": scaler_x.transform(rls_input).numpy()}
# rls_onnx_output = ort_session.run(None, rls_onnx_input)
# rls_output = scaler_y.inverse_transform(rls_onnx_output[0].reshape(-1, 1))

# print("use case\n" + "-" * 20)
# print(f"输入数据：{rls_input}")
# print(f"ONNX 输入：{rls_onnx_input}")
# print(f"ONNX 输出：{rls_onnx_output}")
# print(f"输出数据：{rls_output}")

# --------------------- save usecase --------------------- #
print(">> save use case...", end="")
usecase_path = os.path.join(path_release, "usecase.txt")

save_usecase([90, 41, 31], usecase_path)
save_usecase([68, 46, 83], usecase_path)
save_usecase([19, 45, 80], usecase_path)
save_usecase([1, 64, 58], usecase_path)
save_usecase([16, 0, 2], usecase_path)

# with open(usecase_path, "w") as f:
#     f.write("onnx test case\n")
#     f.write("--------------------\n")
#     f.write(f"input: {rls_input}\n")
#     f.write(f"onnx_input: {rls_onnx_input}\n")
#     f.write(f"onnx_output: {rls_onnx_output}\n")
#     f.write(f"output: {ort_output}\n")
print("done.")

# --------------------- save scaler ---------------------- #
print(">> save scaler...", end="")
scalers_dict = {
    "scaler_in": {
        "mean": scaler_x.mean_.tolist(),
        "var": scaler_x.var_.tolist(),
        "std": scaler_x.scale_.tolist(),
    },
    "scaler_out": {
        "mean": scaler_y.mean_.tolist(),
        "var": scaler_y.var_.tolist(),
        "std": scaler_y.scale_.tolist(),
    },
}

with open(os.path.join(path_release, "scalers_params_dict.yaml"), "w") as f:
    yaml.dump(scalers_dict, f)
print("done.")
print("> release done.")
