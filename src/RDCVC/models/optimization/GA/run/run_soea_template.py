# -*- coding: utf-8 -*-

import geatpy as ea
from matplotlib import pyplot as plt
import datetime

from utils.Ploter import Ploter  # 导入绘图工具
from CoreGA.models.problem_soea_api_SUPP_VolPropRMSE import MyProblem  # 导入自定义问题接口


def generate_filename(problem_name, _NIND=0, MAXGEN=0, MAXTIME=0):
    """生成文件名

    主要功能细节：
        生成文件名，包括问题名称、时间戳


    Args:
        problem_name (str): 问题名称
        _NIND (int): 种群规模
        MAXGEN (int): 最大进化代数
        MAXTIME (int): 最大运行时间 (s)

    Returns:
        filename (str): 文件名
    """
    now_datetime = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    filename = f"{problem_name}_N{_NIND}_G{MAXGEN}_T{MAXTIME}_{now_datetime}"
    return filename


# todo: 脚本运行命令行化
# todo: 并行

if __name__ == '__main__':
    """===============================实例化问题对象==============================="""
    # 实例化问题对象，可选参数：store_data (bool), use_existing_dat
    problem = MyProblem(store_data=True, use_existing_data=False)
    """=================================种群设置================================="""
    NIND = 5  # 种群规模
    Encoding = 'RI'  # 编码方式
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================  构建算法  ================================"""
    algorithm = ea.soea_SEGA_templet(problem,
                                     population,
                                     MAXGEN=2,  # 最大进化代数
                                     MAXTIME=10 * 3600,  # 最大运行时间 (s)
                                     logTras=1,  # 设置每隔多少代记录日志，若设置成 0 则表示不记录日志
                                     verbose=True,  # 设置是否打印输出日志信息
                                     drawing=1,  # 设置绘图方式
                                     aimFuncTrace=True  # 设置是否记录目标函数值的变化
                                     )
    algorithm.outFunc = None  # 设置每次进化记录的输出函数
    print("Algorithm initialized. Start evolution...")
    """===============================  求  解  ================================"""
    save_dirName = f'../checkpoints/{generate_filename(problem.name, NIND, algorithm.MAXGEN, algorithm.MAXTIME)}'
    res = ea.optimize(algorithm,
                      # seed=1,  # 随机数种子
                      prophet=None,  # 先验知识
                      verbose=True,  # 设置是否打印输出日志信息
                      drawing=1,  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
                      outputMsg=True,
                      drawLog=True,
                      saveFlag=True,
                      dirName=save_dirName)
    """=================================输出结果================================="""
    # -----------------------------------
    # 保存进化过程的种群信息 (trace)，图与数据
    # -----------------------------------
    # algorithm.trace 的结构：{'f_best': [], 'f_avg': []}
    Ploter.plot(algorithm.trace, title='Iterative Trace', xlabel='Generation', ylabel='Volume Proportion',
                save_path=save_dirName + '/' + 'trace.png', dpi=300)
    # 保存 trace 原始数据
    with open(save_dirName + '/' + 'trace.txt', 'w') as file:
        file.write(str(algorithm.trace))
    # ================== 保存 result 图 ==================
    # -------------
    # 保存优化后风量图
    # -------------
    opt_Phen = res['Vars'].reshape(-1)  # 优化后的决策变量
    X_opt = {
        "MAU_FREQ": opt_Phen[0] / 10,
        "AHU_FREQ": opt_Phen[1] / 10,
        "EF_FREQ": opt_Phen[2] / 10,
        "RM1_SUPP_DMPR_0": opt_Phen[3],
        "RM2_SUPP_DMPR_0": opt_Phen[4],
        "RM3_SUPP_DMPR_0": opt_Phen[5],
        "RM4_SUPP_DMPR_0": opt_Phen[6],
        "RM5_SUPP_DMPR_0": opt_Phen[7],
        "RM6_SUPP_DMPR_0": opt_Phen[8],
        "RM6_SUPP_DMPR_1": opt_Phen[9]}  # 优化后的决策变量字典
    opt_result = problem.get_result(X_opt)  # 优化后的结果
    volume_key = ["RM1_SUPP_VOL", "RM2_SUPP_VOL", "RM3_SUPP_VOL", "RM4_SUPP_VOL", "RM5_SUPP_VOL", "RM6_SUPP_VOL",
                  "RM7_SUPP_VOL"]
    opt_volume = [opt_result[key] for key in volume_key]  # 优化后的送风量
    res_data = {'ref_volume': problem.calReferObjV(), 'opt_volume': opt_volume}  # 送风量绘图数据
    Ploter.plot(res_data, ['RM1', 'RM2', 'RM3', 'RM4', 'RM5', 'RM6', 'RM7'], xlabel='Room',
                ylabel='Supply Volume (CMH)', size=(6.5, 4),
                save_path=save_dirName + '/' + 'result.png', dpi=300)  # 保存优化后风量图
    plt.show()
    # ================== 输出状态 ==================
    print(f'求解状态：{res["success"]}')
    print(f'评价次数：{res["nfev"]}')
    print(f'时间花费：{res["executeTime"]}秒')
    if res['success']:
        print(f'最优的目标函数值为：{res["ObjV"]}')
        print(f'最优的决策变量值为：{res["Vars"]}')
    else:
        print('此次未找到可行解。')
