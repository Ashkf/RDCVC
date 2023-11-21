import os
import pickle
import yaml

# from sklearn.preprocessing import StandardScaler

# 读取标准化器字典
scalers_dir = r"D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt\checkpoints\NN\dapn12_BS1_LR0.001_EP10000_2023-08-25T16-15-02"
with open(os.path.join(scalers_dir, "x_normalizer.pkl"), "rb") as f:
    scalers_dict = pickle.load(f)

# 输入特征的标准化器
scaler_in = scalers_dict["scaler_in"]
scaler_in_mean = scaler_in.mean_
scaler_in_var = scaler_in.var_
scaler_in_scale = scaler_in.scale_

# 输出标签的标准化器
scaler_out = scalers_dict["scaler_out"]
scaler_out_mean = scaler_out.mean_
scaler_out_var = scaler_out.var_
scaler_out_scale = scaler_out.scale_

# 保存输入输出标准化器参数至 yaml 文件
scalers_dict = {
    "scaler_in": {
        "mean": scaler_in_mean.tolist(),
        "var": scaler_in_var.tolist(),
        "std": scaler_in_scale.tolist(),
    },
    "scaler_out": {
        "mean": scaler_out_mean.tolist(),
        "var": scaler_out_var.tolist(),
        "std": scaler_out_scale.tolist(),
    },
}
with open("scalers_params_dict.yaml", "w") as f:
    yaml.dump(scalers_dict, f)
