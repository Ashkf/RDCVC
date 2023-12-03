"""
* pytorch 中，有关 GPU 的 CUDA 可用性测试
*
* File: test_torch_cuda.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-16 10:24:52
* ----------------------------
* Modified: 2023-11-19 08:48:01
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import torch


def get_device_info():
    is_gpu_available = torch.cuda.is_available()
    device_info = {
        "device": torch.device("cuda:0" if is_gpu_available else "cpu"),
        "device_index": torch.cuda.current_device(),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "is_gpu_available": is_gpu_available,
        "gpu_count": torch.cuda.device_count(),
        "is_bf16_supported": torch.cuda.is_bf16_supported(),
        "device_name": torch.cuda.get_device_name(),
        "device_capability": torch.cuda.get_device_capability(),
        "total_memory_gb": (
            torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024
        ),
        "is_tensorcore_supported": torch.cuda.get_device_properties(0).major >= 7,
        "memory_usage_percentage": (
            torch.cuda.memory_allocated(0)
            / torch.cuda.get_device_properties(0).total_memory
            * 100
        ),
    }
    return device_info


def print_device_info(device_info):
    print(f"当前设备：{device_info['device']}")
    print(f"当前设备索引：{device_info['device_index']}")
    print("CUDA 版本：", device_info["cuda_version"])
    print("Pytorch 版本：", device_info["pytorch_version"])
    print("显卡是否可用：", "可用" if device_info["is_gpu_available"] else "不可用")
    print("显卡数量：", device_info["gpu_count"])
    print(
        "是否支持 BF16 数字格式：",
        "支持" if device_info["is_bf16_supported"] else "不支持",
    )
    print("当前显卡型号：", device_info["device_name"])
    print("当前显卡的 CUDA 算力：", device_info["device_capability"])
    print("当前显卡的总显存：", device_info["total_memory_gb"], "GB")
    print(
        "是否支持 TensorCore:",
        "支持" if device_info["is_tensorcore_supported"] else "不支持",
    )
    print("当前显卡的显存使用率：", device_info["memory_usage_percentage"], "%")


if __name__ == "__main__":
    device_info = get_device_info()
    print_device_info(device_info)
