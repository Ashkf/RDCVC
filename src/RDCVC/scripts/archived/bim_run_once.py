"""
从 bim 中运行一次
"""

import time
import json

from CoreGA.utils.DampersProcessor import DampersProcessor
from CoreGA.utils.DataProcessor import DataProcessor

if __name__ == "__main__":
    """实验并生成数据

    在 bim 中实验并生成数据
    """

    damperProcessor = DampersProcessor()

    start = time.time()  # 计时

    X_test = {"MAU_FREQ": 20.7,
              "AHU_FREQ": 22.2,
              "EF_FREQ": 43.3,
              "RM1_SUPP_DMPR_0": 89,
              "RM2_SUPP_DMPR_0": 86,
              "RM3_SUPP_DMPR_0": 58,
              "RM4_SUPP_DMPR_0": 10,
              "RM5_SUPP_DMPR_0": 57,
              "RM6_SUPP_DMPR_0": 12,
              "RM6_SUPP_DMPR_1": 76,
              "RM2_RET_DMPR_0": 18,
              "RM3_RET_DMPR_0": 82,
              "RM4_RET_DMPR_0": 70,
              "RM6_RET_DMPR_0": 68,
              "RM3_EXH_DMPR_0": 90,
              "RM4_EXH_DMPR_0": 90,
              "RM5_EXH_DMPR_0": 90,
              "RM5_EXH_DMPR_1": 90
              }
    # 计算 bim 模型
    result = DataProcessor.parse_bim_result(damperProcessor.compute(X_test))
    # 解析为 json
    result = json.dumps(result, indent=4)
    print(result)

    end = time.time()
    print("time cost: ", end - start)
