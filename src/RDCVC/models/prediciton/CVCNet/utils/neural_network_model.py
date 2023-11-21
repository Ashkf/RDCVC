"""
neural_network.py

该文件包含了神经网络推理类 NeuralNetwork 的定义。
NeuralNetwork 类是一个抽象类，提供了神经网络推理的通用框架。
具体的神经网络推理类应继承自该类并实现特定的加载模型、预处理和后处理方法。
"""
import os
import pickle

import numpy as np
import torch
from sklearn.discriminant_analysis import StandardScaler
from torch import Tensor
from torch.nn import Module


def load_scaler(scaler_path: str) -> StandardScaler:
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return scaler


class NeuralNetworkModel:
    def __init__(
        self,
        model: Module,
        x_scaler: StandardScaler,
        y_scaler: StandardScaler,
    ):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler

    def forward(self, input_data: Tensor) -> Tensor:
        with torch.no_grad():
            output_tensor = self.model(input_data)
        return output_tensor

    def predict(self, input_data: any):
        preprocessed_input = self._pre_process(input_data)
        output_data = self.forward(preprocessed_input)
        return self._post_process(output_data)

    def predict_pretty(self, input_data: any):
        _result = self.predict(input_data)
        _label_key = [
            "系统送风",
            "系统新风",
            "系统回风",
            "系统排风",
            "一更压差",
            "二更压差",
            "测一压差",
            "测二压差",
            "测三压差",
            "走廊压差",
        ]
        return  {k: _result[:, i] for i, k in enumerate(_label_key)}


    def _pre_process(self, input_data: any) -> Tensor:
        input_data = np.array(input_data, dtype=float).reshape(1, -1)  # to ndarry
        scaled_input = self.x_scaler.transform(input_data)
        return torch.from_numpy(scaled_input).float()  # to tensor

    def _post_process(self, output_data: Tensor) -> any:
        # [:,:10] 是因为模型的一个错误，模型输出维度应该是 10，错设为 11
        _output_data = output_data.detach().numpy().reshape(1, -1)[:, :10] # to ndarray
        return self.y_scaler.inverse_transform(_output_data)



if __name__ == "__main__":
    model_dir = r"D:\OneDrive\01 WORK\# DampersClusterControl\Model-checkpoints\NN_archived\rdc\cmn-mlp\cmn-mlp_6_1000_BS128_LR0.001_EP10000_2023-08-19T17-13-59"
    # model_path = os.path.join(model_dir, "ckps", "ckp_E0157-B0000.pth")
    model_path = os.path.join(model_dir, "final_model.pth")
    model = torch.load(model_path)

    x_scaler_path = os.path.join(model_dir, "x_normalizer.pkl")
    x_scaler = load_scaler(x_scaler_path)

    y_scaler_path = os.path.join(model_dir, "y_normalizer.pkl")
    y_scaler = load_scaler(y_scaler_path)

    model = NeuralNetworkModel(model, x_scaler=x_scaler, y_scaler=y_scaler)
    input_data = {
        "新风机频率": 11.35,
        "送风机频率": 38.49,
        "排风机频率": 22.35,
        # -------------------------- 送风 -------------------------- #
        "一更送风阀": 15,
        "二更送风阀": 35,
        "测一送风阀": 25,
        "测二送风阀": 65,
        "测三送风阀": 50,
        "走廊送风阀 0": 30,
        "走廊送风阀 1": 20,
        # -------------------------- 回风 -------------------------- #
        "二更回风阀": 80,
        "测一回风阀": 10,
        "测二回风阀": 20,
        "走廊回风阀": 60,
        # -------------------------- 排风 -------------------------- #
        "测一排风阀": 80,
        "测二排风阀": 85,
        "测三排风阀": 20,
        "测一排风阀（固）": 60,
    }
    input_data = list(input_data.values())  # to list
    rlt = model.predict_pretty(input_data)
