"""
测试内容包括：
1. 测试数据库是否能够正常连接
2. 测试数据库是否能够正常创建表格
3. 测试数据库是否能够正常 create_table
4. 测试数据库是否能够正常 check_existence
5. 测试数据库是否能够正常 get_data
"""
from CoreGA.utils.DampersProcessor import DampersProcessor
from CoreGA.utils.DataProcessor import DataProcessor
from ...utils.Ploter import Ploter
from utils.SQLiteManager import SQLiteManager


# 预设参数


def init_db(table_name):
    _DBM = SQLiteManager("../data/shuimuBIM.sqlite3")
    _DBM.create_table(table_name)
    return _DBM


def test(_DBM, DP, X: dict, table_name, is_using_existing_data=True, is_storing=True):
    is_exist = 0  # 数据库中是否已经存在该个体的计算结果，0 表示不存在，否则表示索引 ID
    # 若数据库中已经存储了该个体的计算结果，直接从数据库中读取
    if is_using_existing_data:
        is_exist = _DBM.check_existence(X, table_name)
        if is_exist:
            result = _DBM.get_data(table_name, _id=is_exist)
        else:
            # 若数据库中不存在该个体的计算结果，直接计算 bim 模型
            result = DataProcessor.parse_bim_result(DP.compute(X))
    else:
        # 若不使用已经存在的数据，直接计算 bim 模型
        result = DataProcessor.parse_bim_result(DP.compute(X))
    if is_storing and not is_exist:
        # 若需要存储数据，且数据库中不存在该个体的计算结果，则将计算结果存入数据库
        _DBM.add_data(result, table_name=table_name)
    return result


if __name__ == '__main__':
    # 随机一个 np 列表，由 7 个整数组成，每个整数的范围为 0-90，两侧闭区间
    # X_test = np.array([0, 68.0, .0, 53.0, 63.0, 73.0, 83.0])
    # X_test = np.random.randint(0, 90, 7)
    table = "SMKNRDC_bim_data"
    DBM = init_db(table)
    damperProcessor = DampersProcessor()
    # DBM.load_cache_table(table)

    # ====================== 测试数据 ======================
    X_test = {"MAU_FREQ": 49.9,
              "AHU_FREQ": 49.9,
              "EF_FREQ": 19.1,
              "RM1_SUPP_DMPR_0": 90,
              "RM2_SUPP_DMPR_0": 83,
              "RM3_SUPP_DMPR_0": 0,
              "RM4_SUPP_DMPR_0": 0,
              "RM5_SUPP_DMPR_0": 6,
              "RM6_SUPP_DMPR_0": 0,
              "RM6_SUPP_DMPR_1": 0}
    rlt = test(DBM, damperProcessor, X_test, table)
    volume_key = ["RM1_SUPP_VOL", "RM2_SUPP_VOL", "RM3_SUPP_VOL", "RM4_SUPP_VOL", "RM5_SUPP_VOL",
                  "RM6_SUPP_VOL",
                  "RM7_SUPP_VOL"]
    opt_volume = [rlt[key] for key in volume_key]
    # ====================== 设计值 ======================
    ref_volume = {'二更': 331,
                  '测试间三': 722,
                  '洁净走廊': 968,
                  '外走道': 0.0,
                  '测试间一': 689,
                  '一更': 235,
                  '测试间二': 815}
    order = ['一更', '二更', '测试间一', '测试间二', '测试间三', '洁净走廊', '外走道']  # 排序顺序
    ref_volume = DataProcessor.dicts_to_ndarray(ref_volume, order=order)  # 参考风量

    res_data = {'ref_volume': ref_volume, 'opt_volume': opt_volume}
    Ploter.plot(res_data, ['RM1', 'RM2', 'RM3', 'RM4', 'RM5', 'RM6', 'RM7'], xlabel='Room',
                save_path='result.png', ylabel='Supply Volume (CMH)', size=(6.5, 4), dpi=300)
