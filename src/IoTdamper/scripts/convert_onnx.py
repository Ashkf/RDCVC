import os

import torch


# Function to Convert to ONNX
def Convert_ONNX(model):
    # set the model to inference mode
    model.eval()

    # 生成一个随机输入
    dummy_input = torch.randn(1, 3, requires_grad=False)

    # Export the model
    torch.onnx.export(
        model,  # model being run
        # model input (or a tuple for multiple inputs)
        dummy_input,
        "final_model.onnx",  # where to save the model
        export_params=True,
        # store the trained parameter weights inside the model file
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,
        # whether to execute constant folding for optimization
        input_names=["modelInput"],  # the model's input names
        output_names=["modelOutput"],  # the model's output names
        dynamic_axes={
            "modelInput": {0: "batch_size"},  # variable length axes
            "modelOutput": {0: "batch_size"},
        },
    )
    print(" ")
    print("Model has been converted to ONNX")


if __name__ == "__main__":
    nndir = (
        r"D:\OneDrive\01 WORK\# DampersClusterControl\03 IoT damper fitting"
        r"\IoTDamperNN\checkpoints\NN\mlp-l12-dpTv_BS17_LR0.001_EP1000_2023-08-11T11-32-50"
    )
    model_path = os.path.join(nndir, "final_model.pth")
    nnmodel = torch.load(model_path)  # 加载神经网络模型

    # Conversion to ONNX
    Convert_ONNX(nnmodel)
