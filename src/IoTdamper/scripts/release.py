"""
* 
* 
* File: release.py
* Author: Fan Kai
* Soochow University
* Created: 2023-10-09 11:00:54
* ----------------------------
* Modified: 2023-11-02 10:05:32
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""


import torch
import os
import pickle
import yaml
import numpy as np
import onnxruntime
import onnx

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
ckp_dir = r"checkpoints/NN/V5_320250_2023-11-01T09-21-54"

path_release = os.path.join(ckp_dir, "release")
path_onnx = os.path.join(path_release, "final_model.onnx")
os.makedirs(path_release, exist_ok=True)

model_torch = torch.load(os.path.join(ckp_dir, "final_model.pth"))
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
    dummy_input = torch.randn(batch_size, input_shape, requires_grad=False)  # 生成张量
    export_onnx_file = onnx_path  # 目的 ONNX 文件名
    torch.onnx.export(
        model,  # 待转换模型
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

if np.allclose(_dummy_torch_out.detach().numpy(), ort_output, rtol=1e-03, atol=1e-05):
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

rls_input = np.array([[75, 90, 36]]).astype(np.float32)
rls_onnx_input = {"modelInput": scaler_x.transform(rls_input)}
rls_onnx_output = ort_session.run(None, rls_onnx_input)
rls_output = scaler_y.inverse_transform(rls_onnx_output[0].reshape(-1, 1))

print("use case\n" + "-" * 20)
print(f"输入数据：{rls_input}")
print(f"ONNX 输入：{rls_onnx_input}")
print(f"ONNX 输出：{rls_onnx_output}")
print(f"输出数据：{rls_output}")

# --------------------- save usecase --------------------- #
print(">> save use case...", end="")
usecase_path = os.path.join(path_release, "usecase.txt")
with open(usecase_path, "w") as f:
    f.write("onnx test case\n")
    f.write("--------------------\n")
    f.write(f"input: {rls_input}\n")
    f.write(f"onnx_input: {rls_onnx_input}\n")
    f.write(f"onnx_output: {rls_onnx_output}\n")
    f.write(f"output: {ort_output}\n")
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
