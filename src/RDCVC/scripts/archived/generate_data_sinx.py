import torch
import csv
import utils.myUtils as utils

range_left = -10
range_right = 10
num_data = 2000

x = torch.tensor(utils.randfloat(range_left, range_right, num_data))
y = torch.sin(x)
result = torch.stack([x, y], dim=1)
with open(r'../CoreNN/data/sin_func_train.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for i in range(result.size()[0]):
        writer.writerow(result[i, :].tolist())
