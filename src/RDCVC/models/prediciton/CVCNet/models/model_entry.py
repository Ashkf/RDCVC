"""
这里是模型选择的入口，可以通过命令行参数选择不同的模型结构。
import 自己的模型到 model_entry 的字典中
"""

import torch
import torch.nn as nn

from .CVCNet import CVCNet

type2help = {
    "cvcnet": "CVCNet 模型。",
}

type2model = {
    "cvcnet-mtl-mlp": CVCNet,
}


def select_model(model_type: str):
    """入口，选择模型结构"""
    _type = model_type.split("_")

    if _type[0] == "cvcnet-mtl-mlp":
        return type2model[_type[0]](
            backbone_depth=int(_type[1]),
            backbone_dim=int(_type[2]),
            head_depth=int(_type[3]),
            head_dim=int(_type[4]),
        )
    else:
        raise ValueError(
            f"模型类型 {model_type} 不存在，请检查输入的模型类型是否正确。"
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
        "kaiming": nn.init.kaiming_normal_,
        "normal": nn.init.normal_,
        "uniform": nn.init.uniform_,
        "default": None,
    }

    init_method = init_methods[
        args.init_method
    ]  # 获取 args.init_method 对应的初始化方法
    if init_method is not None:
        # 若指定了初始化方式，args.init_method 不为 None，则对模型参数进行指定方法的初始化
        pass
        return model

    # 若 args.init_method 为 None，则使用 PyTorch 默认的初始化方法
    return model


def equip_device(model: torch.nn.Module, device: [str, [int]]):
    """Equip the model with the specified device for processing.

    Args:
        model (torch.nn.Module): The model to be equipped with the device.
        device (List[str, List[int]]): The device to be used for processing.
            The first element of the list should be either 'cpu' or 'cuda'.
            If 'cuda' is chosen, the second element should be a list of GPU device IDs.

    Returns:
        torch.nn.Module: The model equipped with the specified device.

    Raises:
        RuntimeError: If CUDA is not available when 'cuda' is chosen as the device.
        ValueError: If an invalid device type is provided. Supported types are 'cpu' and 'cuda'.
    """
    if device[0] == "cpu":
        model = model.to("cpu")  # CPU processing
    elif device[0] == "cuda":
        if torch.cuda.is_available():
            # Multi-GPU parallel processing
            model = torch.nn.DataParallel(model, device_ids=device[1]).to("cuda")
        else:
            raise RuntimeError("CUDA is not available.")
    else:
        raise ValueError("Invalid device type. Supported types are 'cpu' and 'cuda'.")
    return model
