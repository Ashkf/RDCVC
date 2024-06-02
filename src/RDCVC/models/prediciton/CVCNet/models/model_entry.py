"""
这里是模型选择的入口，可以通过命令行参数选择不同的模型结构。
import 自己的模型到 model_entry 的字典中
"""

import torch
import torch.nn as nn

from .cvcnet import CVCNet
from .IoTDamper_mlp import DAPN12
from .kan_efficient import KANe
from .mmoe import ML_MMoE
from .split import SplitMTL

# from .submodules import DNN

# # todo: 迁移至 README.md
# type2help = {
#     "split-mtl": (
#         "仅在输出层进行多任务划分的模型。Bottom 块为 DNN，输出层为 DNN。"
#         "split-mtl_<inputs_dim>_<bottom_units, 32:32>_"
#     ),
# }

type2model = {
    "dapn12": DAPN12,
    "split-mtl": SplitMTL,
    "cvcnet-mtl-mlp": CVCNet,
    "mmoe-mtl-mlp": ML_MMoE,
    "kane": KANe,
}


def select_model(model_type: str):
    """入口，选择模型结构"""
    _type = model_type.split("_")
    _model_name = _type[0]

    match _model_name:
        case "dapn12":
            return DAPN12()
        case "split-mtl":
            # e.g.: split-mtl_18-32-32_leakyrelu
            return SplitMTL(
                width=[int(v) for v in _type[1].split("-")],
                target_dict={"Airflow": 4, "Pres": 6},
                activation=_type[2],
            )
        case "cvcnet-mtl-mlp":
            # e.g.: cvcnet-mtl-mlp_18_2_5_9_64-64-64_32-32-32
            return CVCNet(
                inputs_dim=int(_type[1]),
                target_dict={"Airflow": 4, "Pres": 6},
                num_layers=int(_type[2]),
                num_tasks_experts=int(_type[3]),
                num_shared_experts=int(_type[4]),
                expert_units=[int(v) for v in (_type[5]).split("-")],
                tower_units=[int(v) for v in _type[6].split("-")],
            )
        case "stl-mlp":
            return
        case "kane":
            # e.g.: kane_1-1-1
            return KANe(layers_hidden=[int(v) for v in _type[1].split("-")])
        case default:
            raise ValueError(
                f"模型类型 {default} 不存在，请检查输入的模型类型是否正确。"
            )


def init_model(model, args, logger):
    """init_model 函数，用于初始化模型。

    在 PyTorch 中，有默认的参数初始化方式。
    因此当定义好网络模型之后，可以不对模型进行显式的参数初始化操作。
    但是，如果想要使用自定义的参数初始化方式，可以使用 torch.nn.init 模块中的函数进行初始化。
    """
    logger.info(f"初始化模型方法：{args.init_method}")
    init_methods = {
        "xavier": nn.init.xavier_normal_,
        "kaiming": _init_model_kaiming,
        "normal": nn.init.normal_,
        "uniform": nn.init.uniform_,
        "default": None,
    }

    init_method = init_methods.get(args.init_method, None)

    # 若 args.init_method 为 None，则使用 kaiming 初始化方法
    if init_method is None:
        init_method = _init_model_kaiming

    return init_method(model)


def _init_model_kaiming(model):
    """使用 kaiming 初始化方法初始化模型参数"""
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            # BN 层的参数初始化
            nn.init.normal_(m.weight, 1, 0.02)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # 全连接层的参数初始化为 kaiming_normal
            nn.init.kaiming_normal_(m.weight, mode="fan_in")
            nn.init.constant_(m.bias, 0)
    return model


def equip_device(model: torch.nn.Module, device: str):
    """Equip the model with the specified device for processing.

    Args:
        model (torch.nn.Module): The model to be equipped with the device.
        device (str): The device to be used for processing.

    Returns:
        torch.nn.Module: The model equipped with the specified device.

    Raises:
        RuntimeError: If CUDA is not available when 'cuda' is chosen as the device.
        ValueError: If an invalid device type is provided. Supported types are 'cpu' and 'cuda'.
    """
    if device == "cpu":
        model = model.to("cpu")  # CPU processing
    elif device == "cuda":
        if torch.cuda.is_available():
            model = model.to("cuda")
        else:
            raise RuntimeError("CUDA is not available.")
    else:
        raise ValueError("Invalid device type. Supported types are 'cpu' and 'cuda'.")
    return model
