"""
*
*
* File: test.py
* Author: Fan Kai
* Soochow University
* Created: 2024-07-01 14:45:49
* ----------------------------
* Modified: 2024-07-01 14:49:53
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

# -*- coding: utf-8 -*-

import geatpy as ea  # import geatpy

if __name__ == "__main__":
    problem = ea.benchmarks.DTLZ1()  # 生成问题对象
    # 构建算法
    algorithm = ea.moea_NSGA3_templet(
        problem,
        ea.Population(Encoding="RI", NIND=100),
        MAXGEN=500,  # 最大进化代数。
        logTras=1,
    )  # 表示每隔多少代记录一次日志信息，0 表示不记录。
    # 求解
    res = ea.optimize(
        algorithm,
        verbose=True,
        drawing=0,
        outputMsg=True,
        drawLog=True,
        saveFlag=True,
        dirName="result",
    )
    print(res)
