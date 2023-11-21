"""
* 
* 
* File: SQLiteManager.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-15 03:38:51
* ----------------------------
* Modified: 2023-11-20 02:21:07
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""


import datetime
import sqlite3

import pandas as pd
from pandas import DataFrame


class SQLiteManager:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self.cache: DataFrame = DataFrame()

    def load_table_cache(self, table_name: str):
        """加载整表数据作为缓存

            todo: 可能需要考虑缓存的大小，以及缓存的有效期

        用 DataFrame 存储，以便于后续的数据处理
        列名与数据库字段名一致

        Args:
            table_name (str): 数据表名称
        """
        self.cache = pd.read_sql(
            f"SELECT * FROM {table_name}", self.conn
        )  # 读取数据库中的数据
        # 若 cache 不为空，则将字段名作为列名
        if not self.cache.empty:
            self.cache.set_index("ID", inplace=True)  # 将 ID 字段设置为索引
        return self.cache

    def check_existence(self, value: dict, table_name: str) -> int:
        """检查数据表是否存在相同的数据

        若存在则返回主键 ID，否则返回 0
        存在的标准：风阀角度相同，风机频率相同，且数据来源相同
            包括以下字段：
                DATA_SRC
                MAU_FREQ
                AHU_FREQ
                EF_FREQ
                RM1_SUPP_DMPR_0
                RM2_SUPP_DMPR_0
                RM3_SUPP_DMPR_0
                RM4_SUPP_DMPR_0
                RM5_SUPP_DMPR_0
                RM6_SUPP_DMPR_0
                RM6_SUPP_DMPR_1
                RM2_RET_DMPR_0
                RM3_RET_DMPR_0
                RM4_RET_DMPR_0
                RM6_RET_DMPR_0
                RM3_EXH_DMPR_0
                RM4_EXH_DMPR_0
                RM5_EXH_DMPR_0
                RM5_EXH_DMPR_1

        Args:
            table_name (str): 数据表名称
            value (list): 个体的决策变量

        Returns:
            ID (int): 主键 ID
        """
        ID = 0  # 默认返回 0，表示不存在相同的数据。若存在相同的数据，则返回主键 ID

        # 检查缓存是否为空
        if self.cache.empty:
            return ID

        # 从缓存中以字典的形式读取数据
        # 'records'：以行索引为键生成列表，以每行数据构成的字典作为值，字典的键为列名。
        # todo: 此处耗时长，可以考虑使用 DataFrame 的 iterrows 方法，以提高效率
        for index, row in self.cache.iterrows():
            # 若 value 的键均在 row 中，且值相同，则返回 ID
            _is_same = True  # 默认为 True，若有不同的值，则为 False
            column_names = row.index.tolist()  # 获取列名
            for key in value:
                # 若 value 的键不在 row 中，则直接跳出循环
                if key not in column_names:
                    _is_same = False
                    break
                # 若 value 的键在 row 中，但值不同，则直接跳出循环
                if value.get(key) != row[key]:
                    _is_same = False
                    break
            # 若 value 的键均在 row 中，且值相同，则返回 ID
            if _is_same:
                ID = index
                print(f"ID 为 {ID} 的数据的从数据库中表 {table_name} 读取。")
                break

        return ID

    def create_table(self, table_name: str):
        """从数据表模板创建数据表

        Args:
            table_name (str): 数据表名称
        """
        template = {
            "SMKNRDC_bim_data": self.init_table_SMKNRDC,
            "default_table": self.init_table,
            "bim2NN_train_data": self.init_table_SMKNRDC,
            "bim2NN_test_data": self.init_table_SMKNRDC,
            "bim2NN_eval_data": self.init_table_SMKNRDC,
            "bim_raw": self.init_table_SMKNRDC,
        }
        if table_name in template:
            template[table_name](table_name)

    def init_table(self, table_name: str):
        """默认的数据表模板"""
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name}(
                ID  INTEGER PRIMARY KEY AUTOINCREMENT,
                A   TEXT,   -- A
                B   TEXT,   -- B
                V   REAL)""")
        self.conn.commit()

    def init_table_SMKNRDC(self, table_name: str):
        """初始化数据表，如果数据表不存在则创建数据表

        表的模板针对于水木科能的研发洁净室的 BIM 模型计算数据
        表名：SMKNRDC_bim_data
            释义：SMKNRDC = ShuiMuKeNeng Research and Development Clean Room

        表的字段：请查看具体 SQL 语句
        """
        self.cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name}(
                ID              INTEGER PRIMARY KEY AUTOINCREMENT,
                DATA_SRC        TEXT    NOT NULL,   -- 数据来源
                COLL_TIME       TEXT,               -- 数据采集时间
                UPD_TIME        TEXT,               -- 数据更新时间
                MAU_FREQ        REAL    NOT NULL,   -- 新风机频率
                AHU_FREQ        REAL    NOT NULL,   -- 送风机频率
                EF_FREQ         REAL    NOT NULL,   -- 排风机频率
                -- 送风阀角度
                RM1_SUPP_DMPR_0 REAL    NOT NULL,
                RM2_SUPP_DMPR_0 REAL    NOT NULL,
                RM3_SUPP_DMPR_0 REAL    NOT NULL,
                RM4_SUPP_DMPR_0 REAL    NOT NULL,
                RM5_SUPP_DMPR_0 REAL    NOT NULL,
                RM6_SUPP_DMPR_0 REAL    NOT NULL,
                RM6_SUPP_DMPR_1 REAL    NOT NULL,
                -- 回风阀角度
                RM2_RET_DMPR_0  REAL    NOT NULL,
                RM3_RET_DMPR_0  REAL    NOT NULL,
                RM4_RET_DMPR_0  REAL    NOT NULL,
                RM6_RET_DMPR_0  REAL    NOT NULL,
                -- 排风阀角度
                RM3_EXH_DMPR_0  REAL    NOT NULL,
                RM4_EXH_DMPR_0  REAL    NOT NULL,
                RM5_EXH_DMPR_0  REAL    NOT NULL,
                RM5_EXH_DMPR_1  REAL    NOT NULL,
                -- 房间 1（一更）状态
                RM1_DIFF_VOL   REAL,
                RM1_EXH_VOL    REAL,
                RM1_PRES       REAL,
                RM1_RET_VOL    REAL,
                RM1_SUPP_VOL   REAL,
                -- 房间 2（二更）状态
                RM2_DIFF_VOL   REAL,
                RM2_EXH_VOL    REAL,
                RM2_PRES       REAL,
                RM2_RET_VOL    REAL,
                RM2_SUPP_VOL   REAL,
                -- 房间 3（测试间一）状态
                RM3_DIFF_VOL   REAL,
                RM3_EXH_VOL    REAL,
                RM3_PRES       REAL,
                RM3_RET_VOL    REAL,
                RM3_SUPP_VOL   REAL,
                -- 房间 4（测试间二）状态
                RM4_DIFF_VOL   REAL,
                RM4_EXH_VOL    REAL,
                RM4_PRES       REAL,
                RM4_RET_VOL    REAL,
                RM4_SUPP_VOL   REAL,
                -- 房间 5（测试间三）状态
                RM5_DIFF_VOL   REAL,
                RM5_EXH_VOL    REAL,
                RM5_PRES       REAL,
                RM5_RET_VOL    REAL,
                RM5_SUPP_VOL   REAL,
                -- 房间 6（洁净走廊）状态
                RM6_DIFF_VOL   REAL,
                RM6_EXH_VOL    REAL,
                RM6_PRES       REAL,
                RM6_RET_VOL    REAL,
                RM6_SUPP_VOL   REAL,
                -- 房间 7（外走道）状态
                RM7_DIFF_VOL   REAL,
                RM7_EXH_VOL    REAL,
                RM7_PRES       REAL,
                RM7_RET_VOL    REAL,
                RM7_SUPP_VOL   REAL,
                -- 系统状态
                TOT_EXH_VOL     REAL,
                TOT_FRSH_VOL    REAL,
                TOT_RET_VOL     REAL,
                TOT_SUPP_VOL    REAL)""")
        self.conn.commit()

    def add_data(self, data: list | dict, table_name="data"):
        """
        添加数据到表。依据 data 的类型判别是添加一条还是添加多条。
            如果是 list 则批量添加，如果是 dict 则单条添加。

        添加时会自动补充一个字段 COLL_TIME，值为当前时间，格式为 ISO 8601。
        """
        if isinstance(data, dict):
            self.add_data_once(data, table_name)  # 添加一条
        elif isinstance(data, list):
            self.add_data_batch(data, table_name)  # 添加多条
        elif data is None:
            print("data is None, cannot add data to table, jump.")
        else:
            raise TypeError("data should be list or dict.")

    def add_data_batch(self, data: list, table_name="data"):
        """批量添加数据到表，每条自动补充一个字段 COLL_TIME，值为当前时间，格式为 ISO 8601"""
        for item in data:
            item["COLL_TIME"] = datetime.datetime.now().isoformat()
        keys = ", ".join(data[0].keys())
        values = ", ".join(["?"] * len(data[0]))

        sql = f"INSERT INTO {table_name} ({keys}) VALUES ({values})"

        try:
            self.cursor.executemany(sql, [tuple(item.values()) for item in data])
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def add_data_once(self, data: dict, table_name="data"):
        """添加一条数据到表，自动补充一个字段 COLL_TIME，值为当前时间，格式为 ISO 8601"""
        data["COLL_TIME"] = datetime.datetime.now().isoformat()
        keys = ", ".join(data.keys())
        values = ", ".join(["?"] * len(data))

        sql = f"INSERT INTO {table_name} ({keys}) VALUES ({values})"

        try:
            self.cursor.execute(sql, tuple(data.values()))
            self.conn.commit()
        except Exception as e:
            print(e)
            self.conn.rollback()

    def get_data(
        self,
        table_name,
        id_range: tuple = None,
        column_name: str = None,
        _id: int = None,
    ):
        """获取数据。

        若传入一个 ID 范围和 column_name，则取回对应行列的数据。
        若传入一个 ID 范围，则取回对应范围所有列的数据。
        若传入一个 ID 和 column_name，则取回该 ID 对应行列的数据。
        若传入一个 ID，则取回该 ID 对应行的所有数据。

        Examples:
            # 获取 id 在 1 到 10 范围内的所有列的数据
            data = get_data_from_sqlite('my_table', id_range=(1, 10))
            # 获取 id 为 1 的某一列的数据
            data = get_data_from_sqlite('my_table', id=1, column_name='my_column')


        Args:
            table_name: 表名
            id_range: ID 范围
            column_name: 列名
            _id: ID
        Returns:
            data(list): 数据
        """
        if column_name is None:
            column_name = "*"

        if id_range is not None:
            if _id is not None:
                raise ValueError(
                    "Either id_range or id should be provided, but not both."
                )
            sql = f"SELECT {column_name} FROM {table_name} WHERE id >= ? AND id <= ?"
            self.cursor.execute(sql, id_range)
        elif _id is not None:
            sql = f"SELECT {column_name} FROM {table_name} WHERE id = ?"
            self.cursor.execute(sql, (_id,))
        else:
            raise ValueError("Either id_range or id should be provided.")
        data = self.cursor.fetchall()

        return data

    def get_all_data(self, table_name):
        sql = f"SELECT * FROM {table_name}"
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        return data

    def get_data_count(self, table_name):
        """获取表中数据条数"""
        sql = f"SELECT count(*) FROM {table_name}"
        self.cursor.execute(sql)
        data = self.cursor.fetchall()
        return data[0][0]

    def update_data(self):
        """更新数据

        记得更新 UPD_TIME 字段
        """
        pass

    def view_data(self):
        pass

    def search_data(
        self, experiment_name="", experiment_date="", experiment_description=""
    ):
        pass

    def delete_data(self):
        pass

    def __del__(self):
        self.conn.close()
