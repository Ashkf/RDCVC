"""
    实际使用时需要将训练好的模型上在输入数据上运行，这里以测试集的数据为例
    
    torch.no_grad() 
        停止 autograd 模块的工作，不计算和储存梯度，一般在用训练好的模型跑测试集时使用，
        因为测试集时不需要计算梯度更不会更新梯度。使用后可以加速计算时间，节约 gpu 的显存
"""

import sys

sys.path.append('.')

from datasets import sindataset as data_loader
from models.base import mlp as my_model
import torch
import torch.utils.data as torchData
from matplotlib import pyplot as plt

model = my_model.MLP_L4()

test_dataset = data_loader.sinDataset(
    file_path='../data/tri-func_2.csv', is_train=False)
test_loader = torchData.DataLoader(dataset=test_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=10)

# 加载训练好的模型

model.load_state_dict(torch.load(
    '../checkpoints/mlp_1_16_16_1_20230225_235257.pth'))
model.eval()

# RUN
if __name__ == '__main__':
    ks, vs, predses = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            k, v = batch[0], batch[1]
            preds = model(k)
            ks.append(k)
            vs.append(v)
            predses.append(preds)

    ks = torch.tensor(ks)
    vs = torch.tensor(vs)
    predses = torch.tensor(predses)

    plt.figure()

    plt.plot(ks, torch.sin(ks), 'b', label='k-sin(x)')
    plt.plot(ks, predses, 'g', label='k-preds')
    # plt.scatter(ks, err)
    # plt.title('err')
    plt.legend()
    plt.tight_layout()

    plt.show()
