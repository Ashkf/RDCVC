# 物联网风阀风量拟合神经网络流程

This repository contains the code for the IoT-Damper neural network airflow fitting pipeline. The pipeline is designed to train a neural network estimator that can predict damper airflow. The neural network is trained using the Pytorch library.

## overview

模型输入是一个包含以下特征的向量：

1. $\theta_f$: 前阀片开度
2. $\theta_b$: 后阀片开度
3. $p$: 风阀读取压差（Pa）

模型输出是一个包含预测风量 $q$ 的单元素向量。

## Data

调用脚本 `intergrate_data.py` 将原始 excel 数据文件汇总，并划分为训练集和测试集。

## Training

训练使用 excel 中的数据集，目的是建立 $g: [\theta_f,\theta_b,p]\rightarrow q$

使用 `train.py` 调用 `model.py` 中的模型进行训练。训练参数于 `.vscode/launch.json` 中设置，也可以执行命令：

```bash
train.py dapn12 iotdp --train_path data/IoT-Damper_v5_320250/IoTDamper.xlsx --eval_path data/IoT-Damper_v5_320250/IoTDamper.xlsx --lr 1e-3 --batch_size 8 --epochs 10000 --save_every 100 --normalize_target x --normalize_method zscore --earlystop --espatience 100 --lr_scheduler 
```

> 注意修改 `train.py` 中的 `--train_path` 和 `--eval_path` 等参数。

训练结果输出在 `checkpoints/NN/` 下，包括模型权重和训练日志。每次训练会生成一个新的目录。

目录下文件清单：

|        Name        |           Description            |
| :----------------: | :------------------------------: |
|       `ckps`       |          训练中的检查点          |
|    `args.yaml`     |             训练参数             |
|  `events.out...`   |         Tensorboard 日志         |
| `final_model.pth`  | 最终模型权重（仅完成训练后生成） |
|     `log_...`      |           训练日志文件           |
| `x_normalizer.pkl` |           特征归一化器           |

## Release

调用 `release.py` 将训练好的模型转换为 ONNX 格式，同时将归一化器保存为 yaml 文件，并保存测试用例。
所有文件保存在训练结果输出的 `checkpoints/NN/**/release/` 目录下。

输出文件清单：

|            Name            |   Description    |
| :------------------------: | :--------------: |
|     `final_model.onnx`     |  ONNX 格式模型   |
| `scalers_params_dict.yaml` | 归一化器参数文件 |
|       `usecase.txt`        |     测试用例     |
