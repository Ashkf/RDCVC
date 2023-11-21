import torch
import csv

PI = torch.pi
range_left = -10
range_right = 10

x = torch.tensor([num * PI / 4 for num in range(range_left, range_right)])
y = torch.sin(x)
result = torch.stack([x, y], dim=1)

with open('data/sin_2.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y'])
    for i in range(result.size()[0]):
        writer.writerow(result[i, :].tolist())
