from utils.DampersProcessor import DampersProcessor  # bim 模型风阀操控接口

X = [0, 0, 0, 0, 0, 0, 0]

dp = DampersProcessor()
volumes = dp.compute_supply_volume(X)
print(volumes)
