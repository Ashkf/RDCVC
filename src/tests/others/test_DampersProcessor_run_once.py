import matplotlib.pyplot as plt
import numpy as np

from utils.DampersProcessor import DampersProcessor
from utils.DataProcessor import DataProcessor
from sklearn.metrics import mean_squared_error as MSE

dp = DampersProcessor()
dp.get_auth_token()  # 获取用户 token
new_damper_config = [
    {'id': '1087931244907331584', 'angle': 28.0, 'damper_name': '一更-SFF-01'},
    {'id': '1087931244907331591', 'angle': 5.0, 'damper_name': '二更-SFF-01'},
    {'id': '1087931244907331588', 'angle': 20.0, 'damper_name': '测试间一-SFF-01'},
    {'id': '1087931244907331586', 'angle': 90.0, 'damper_name': '测试间二-SFF-01'},
    {'id': '1087931244907331587', 'angle': 88.0, 'damper_name': '测试间三-SFF-01'},
    {'id': '1087931244903137285', 'angle': 87.0, 'damper_name': '洁净走廊-SFF-01'},
    {'id': '1087931244907331589', 'angle': 42.0, 'damper_name': '洁净走廊-SFF-02'}]

dp.set_dampers(new_damper_config)  # 设置阀门角度
dp.update_impedance(dp.preset["requestSupplySystemId"])  # 更新阻抗
result = dp.compute_sync()  # 计算
# now_time = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')  # 获取当前时间
# json.dump(result, open(f"result_{now_time}.json", "w"))  # 保存结果

real_supply_volumes = result["roomNameToSupplyWindVolume"]  # 实际送风量
design_supply_volumes = dp.ref_config  # 设计送风量

real_supply_volumes = DataProcessor.dicts_to_ndarray(real_supply_volumes, order=dp.order)
design_supply_volumes = DataProcessor.dicts_to_ndarray(design_supply_volumes, order=dp.order)
real_supply_p = DataProcessor.cal_proportions(real_supply_volumes)  # 计算实际送风量占比
design_supply_p = DataProcessor.cal_proportions(design_supply_volumes)  # 计算设计送风量占比

print(f"R2: {[damper['angle'] for damper in new_damper_config]} -> {np.sqrt(MSE(real_supply_p, design_supply_p))}")

# 绘制实际风量与设计风量的折线图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 5))
plt.plot(dp.order, real_supply_p, "bd-", label="bim")
plt.plot(dp.order, design_supply_p, "rs--", label="design")
plt.legend()
plt.show()
