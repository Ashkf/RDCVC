import matplotlib.pyplot as plt
import pandas as pd
import scienceplots

df_1 = pd.read_csv(
    r"C:\Users\KAI\Desktop\run-NN_mlp-L12_BS128_LR0.001_EP1000_2023-05-27T11-33-13-tag-val_room_pres_mae.csv")
df_2 = pd.read_csv(
    r"C:\Users\KAI\Desktop\run-NN_mlp-L12_resume_BS128_LR0.001_EP2000_2023-05-28T15-32-41-tag-val_room_pres_mae.csv")
df_val = pd.concat([df_1, df_2], ignore_index=True)  # 合并两个 DataFrame
df_val.sort_values(by="Step", inplace=True, ascending=True)

df_1 = pd.read_csv(
    r"C:\Users\KAI\Desktop\run-NN_mlp-L12_BS128_LR0.001_EP1000_2023-05-27T11-33-13-tag-train_room_pres_mae.csv")
df_2 = pd.read_csv(
    r"C:\Users\KAI\Desktop\run-NN_mlp-L12_resume_BS128_LR0.001_EP2000_2023-05-28T15-32-41-tag-train_room_pres_mae.csv")
df_train = pd.concat([df_1, df_2], ignore_index=True)  # 合并两个 DataFrame
df_train.sort_values(by="Step", inplace=True, ascending=True)

loss = {
    'val/pres_mae': df_val["Value"].tolist(),
    'train/pres_mae': df_train["Value"].tolist()
}
with plt.style.context(['science', 'grid', 'no-latex']):
    fig, ax = plt.subplots()
    ax.plot(df_val["Value"].tolist(), label="val/pres_mae")
    ax.plot(df_train["Value"].tolist(), linestyle='-.', label="train/pres_mae")
    plt.title("Room Pressure Error")  # 图标题
    ax.legend()  # 图例标题
    ax.set(xlabel="Epoch")  # X 轴标题
    ax.set(ylabel="Pressure (Pa)")  # Y 轴标题
    # ax.autoscale(tight=True)
    # ax.set_ylim(top=3) # 设置 Y 刻度最大值

    save_path = r"C:\Users\KAI\Desktop\room_pres_mae.png"
    fig.savefig(save_path, dpi=300, bbox_inches='tight')