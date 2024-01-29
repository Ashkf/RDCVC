"""
检查内容：check_existence
    1. 从数据库中读取整张表缓存
    2. 使用缓存加速测试
"""
from utils.SQLiteManager import SQLiteManager
import time


def init_db(table_name: str, database_path: str = "../data/shuimuBIM.sqlite3"):
    # 初始化数据库
    _DBM = SQLiteManager(database_path)
    _DBM.create_table(table_name)
    return _DBM


if __name__ == '__main__':
    X_test = {
        "RM1_SUPP_DMPR_0": 54,
        "RM2_SUPP_DMPR_0": 88,
        "RM3_SUPP_DMPR_0": 73,
        "RM4_SUPP_DMPR_0": 58,
        "RM5_SUPP_DMPR_0": 3,
        "RM6_SUPP_DMPR_0": 74,
        "RM6_SUPP_DMPR_1": 2
    }
    # X_test = [90, 90, 90, 90, 90, 90, 90]
    answer = 5

    table = "SMKNRDC_bim_data"
    DBM = init_db(table)
    DBM.load_table_cache(table)
    start = time.time()  # 计时
    for i in range(100):
        DBM.check_existence(X_test, table)
    end = time.time()
    print(f'耗时：{end - start} s')
