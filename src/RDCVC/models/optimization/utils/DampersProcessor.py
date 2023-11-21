import datetime
import sys
import time
import numpy as np
import requests
import json

from CoreGA.utils.APIError import APIError
from CoreGA.utils.DataProcessor import DataProcessor


class DampersProcessor:
    """该类用于处理风阀的操控，与 BIM 模型进行交互
    # todo: 更名为 BIMProcessor
    Attributes:
        dampers (dict): 风阀配置字典，键为风阀 UID，值为风阀配置
        auth_token (str): 用户 authorization token，通过登录方法获取
        order (list): 房间排序顺序
        preset (dict): 预设参数
    """

    def __init__(self, order=None):
        """
        Args:
            order (): 房间排序顺序
        """
        # self.dampers = {}  # 风阀配置字典，修正后替代 dampers 属性
        self.damper_keys = ["RM1_SUPP_DMPR_0",
                            "RM2_SUPP_DMPR_0",
                            "RM3_SUPP_DMPR_0",
                            "RM4_SUPP_DMPR_0",
                            "RM5_SUPP_DMPR_0",
                            "RM6_SUPP_DMPR_0",
                            "RM6_SUPP_DMPR_1",
                            "RM2_RET_DMPR_0",
                            "RM3_RET_DMPR_0",
                            "RM4_RET_DMPR_0",
                            "RM6_RET_DMPR_0",
                            "RM3_EXH_DMPR_0",
                            "RM4_EXH_DMPR_0",
                            "RM5_EXH_DMPR_0",
                            "RM5_EXH_DMPR_1"]  # 风阀配置存在数据库中的字段名
        self.fan_keys = ["MAU_FREQ", "AHU_FREQ", "EF_FREQ"]  # 风机配置存在数据库中的字段名
        self.auth_token = ""

        self.order = order or ['一更', '二更', '测试间一', '测试间二', '测试间三', '洁净走廊', '外走道']
        self.preset = {
            "userid": "1398",
            "projectId": "6437632d05cab969b48d8890",  # 动态值
            "customerAccount": "szsmkn",
            "identityType": "normal",
            "identifier": "fk",
            "credential": "123456",
            "url_prefix": "http://airapi.enercotech.com",
            # "url_prefix": "http://airductapiv2.ruifangs.com",
            "requestSupplySystemId": "1125084748981469234",  # 动态值
            "requestReturnSystemId": "1125084748998246462",  # 动态值
            "requestExhaustSystemId": "1125084748973080576",  # 动态值
            "relatedExhaustFanId": "6332b25ae4b02e03a49e66da",  # 动态值
            "relatedNewFanId": "6332b424e4b02e03a49e66db",  # 动态值
            "relatedSupplyFanId": "63b3904ce4b07136339a4182"  # 动态值
        }
        self.get_auth_token()  # 获取用户 authorization token
        # self.get_system_ids()  # 获取系统 id
        # self.get_dampers()  # 获取风阀配置，未嵌入到类中，若变更系统则可以参考此方法

    def get_auth_token(self):
        """获取用户 authorization token"""
        url = self.preset["url_prefix"] + "/login"
        payload = json.dumps({
            "customerAccount": self.preset["customerAccount"],
            "identityType": self.preset["identityType"],
            "identifier": self.preset["identifier"],
            "credential": self.preset["credential"]
        })
        headers = {
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
            'Content-Type': 'application/json'
        }

        rq = self.request_securely(method="POST",
                                   url=url,
                                   headers=headers,
                                   data=payload)
        if (rq is None) or (rq.status_code != 200):
            try:
                # 两种情况：1. 请求失败；2. 请求成功但是获取 token 失败
                raise APIError(f"Failed to get auth token, Check your network connection.")
            except Exception as e:
                print(f'进程退出，错误信息：{e}')
                sys.exit(1)

        self.auth_token = rq.text

    def get_damper_configs(self, system_id):
        """从系统 id 及用户 token 获取风阀配置"""
        url = self.preset["url_prefix"] + f"/v2/systems/{system_id}/damper-configs"

        payload = {}
        headers = {
            'Authorization': f'{self.auth_token}',
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
        }

        response = requests.request("GET", url, headers=headers, data=payload, proxies={})
        return response.text

    def prepare_damper_config(self, config) -> dict:
        """
        准备单个风阀开度配置

        Returns:
            config(dict): 风阀开度配置的 json 字符串
        """
        config = {
            "angle": config["angle"],
            "id": config["id"],
            "userId": self.preset["userid"]
        }
        return config

    def set_fan(self, fan_config):
        """设置风机频率

        api: /v2/fan-configs

        Args:
            fan_config(dict): 风机配置字典
        """
        url = self.preset["url_prefix"] + "/v2/fan-configs"  # 设置风阀群角度的 api
        # todo: FanId 应该自动获取
        payload = json.dumps({
            "actualExhaustFrequency": fan_config["EF_FREQ"],
            "actualNewFrequency": fan_config["MAU_FREQ"],
            "actualSupplyFrequency": fan_config["AHU_FREQ"],
            "id": "1125084819399639040",  # 动态值
            "projectId": self.preset["projectId"],  # 动态值
            "newFanExist": "true",
            "relatedExhaustFanId": self.preset["relatedExhaustFanId"],
            "relatedNewFanId": self.preset["relatedNewFanId"],
            "relatedSupplyFanId": self.preset["relatedSupplyFanId"],
            "userId": self.preset["userid"]
        })
        headers = {
            'Authorization': self.auth_token,
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
            'Content-Type': 'application/json'
        }

        rq = self.request_securely(method="POST",
                                   url=url,
                                   headers=headers,
                                   data=payload)
        if (rq is None) or (rq.status_code != 200):
            # 两种情况：1. 请求失败；2. 请求成功但是设定风阀失败
            raise APIError(f"Failed to set fans.")

    def set_dampers(self, dampers_config):
        """设置风阀角度

        api: /v2/damper-configs/batch

        Args:
            dampers_config(list): 风阀配置列表，列表中的每个元素是一个配置字典
        """
        data = []
        url = self.preset["url_prefix"] + "/v2/damper-configs/batch"  # 设置风阀群角度的 api
        for i in range(len(dampers_config)):
            # dampers_set 是一个列表，列表中的每个元素是一个配置字典
            data.append(self.prepare_damper_config(dampers_config[i]))
        # 风阀群配置的 json 字符串，同时将 numpy.int32 转换为 int
        payload = json.dumps(data, default=lambda o: int(o) if isinstance(o, np.int32) else o)
        headers = {
            'Authorization': f'{self.auth_token}',
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
            'Content-Type': 'application/json'
        }

        rq = self.request_securely(method="POST",
                                   url=url,
                                   headers=headers,
                                   data=payload)
        if (rq is None) or (rq.status_code != 200):
            # 两种情况：1. 请求失败；2. 请求成功但是设定风阀失败
            raise APIError(f"Failed to set dampers.")

    def print_damper_configs(self):
        """打印风阀配置"""
        configs = self.get_damper_configs(self.preset["requestSupplySystemId"])
        dco = json.loads(configs)
        for i in range(len(dco)):
            print(f"{i:02}: {dco[i]['id']}, {dco[i]['angle']}")

    def update_impedance(self, system_id):
        """
        更新阻抗

        api: /v2/systems/{systemId}/update-impedance

        Args:
            system_id (str): 系统 id
        """
        url = self.preset["url_prefix"] + f"/v2/systems/{system_id}/update-impedance"

        payload = {}
        headers = {
            'Authorization': f'{self.auth_token}',
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
        }

        rq = self.request_securely(method="POST",
                                   url=url,
                                   headers=headers,
                                   data=payload)
        if (rq is None) or (rq.status_code != 200):
            # 两种情况：1. 请求失败；2. 请求成功但是更新阻抗失败
            raise APIError(f"Failed to update impedance.")

    @staticmethod
    def request_securely(method, url, headers, data, max_retry=5, retry_wait=3) -> requests.Response | None:
        """安全地请求

        Args:
            method (str): 请求方法
            url (str): 请求 url
            headers (dict): 请求头
            data (str|dict): 请求体
            max_retry (int): 最大重试次数
            retry_wait (int): 重试等待时间

        Returns:
            response (requests.Response|None): 请求响应

        Raises:
            Exception: 请求失败
        """
        is_retry = False  # 是否属于重试
        for i in range(max_retry):
            try:
                if is_retry:
                    print(f"正在重试...")
                response = requests.request(method, url, headers=headers, data=data, proxies={})
                if response.status_code != 200:
                    raise Exception(f"Failed to request, status code: {response.status_code} ")
                if is_retry:
                    print("成功！")
                return response  # 正常返回
            except requests.exceptions.Timeout:
                print(f"连接超时，等待{retry_wait}s 后重试...({i + 1}/{max_retry})", end="")
                is_retry = True
                time.sleep(retry_wait)
            except requests.exceptions.ConnectionError:
                print(f"连接错误，等待{retry_wait}s 后重试...({i + 1}/{max_retry})", end="")
                is_retry = True
                time.sleep(retry_wait)
            except Exception as e:
                print(f"意外的错误！{e}，等待 {retry_wait}s 后重试...({i + 1}/{max_retry})", end="")
                is_retry = True
                time.sleep(retry_wait)

        print(f"重试 {max_retry} 次后仍然失败，请检查网络连接！")
        return None  # 重试失败

    def compute_sync(self) -> dict:
        """
        同步整体计算 by projectId

        api: /v2/sync/calculate-airduct

        Returns:
            result_json (dict): 计算结果
        """
        url = self.preset["url_prefix"] + "/v2/sync/calculate-airduct"

        payload = json.dumps({
            "projectId": f"{self.preset['projectId']}",
            "requestExhaustSystemId": f"{self.preset['requestExhaustSystemId']}",
            "requestReturnSystemId": f"{self.preset['requestReturnSystemId']}",
            "requestSupplySystemId": f"{self.preset['requestSupplySystemId']}"
        })
        headers = {
            'Authorization': f'{self.auth_token}',
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)',
            'Content-Type': 'application/json'
        }

        rq = self.request_securely(method="POST",
                                   url=url,
                                   headers=headers,
                                   data=payload)
        if (rq is None) or (rq.status_code != 200):
            # 两种情况：1. 请求失败；2. 请求成功但是计算失败
            raise APIError(f"Failed to calculation sync.")
        result_json = json.loads(rq.text)
        return result_json

    def compute(self, X):
        """
        传入参数设定字典 --计算-> 实验室整体状态

        计算逻辑：
            1. 设置参数
            2. 更新阻抗
            3. 同步计算

        Args:
            X (dict): 参数字典

        Returns:
            _result (dict): 计算结果
        """

        if not isinstance(X, dict):  # 检查：X 应为 dict
            raise TypeError(f"X 应为 dict, but got {type(X)}")

        try:
            # 设置风阀
            if any([key in X for key in self.damper_keys]):
                for key in X:
                    if key in self.damper_keys:
                        if not (0 <= X[key] <= 90):
                            raise ValueError(f"风阀角度应在 [0, 90], but got {X[key]}")
                self.set_dampers([
                    {'id': "1125084774763855883",
                     'angle': X["RM1_SUPP_DMPR_0"] if X.get("RM1_SUPP_DMPR_0") is not None else 0},
                    {'id': "1125084774763855877",
                     'angle': X["RM2_SUPP_DMPR_0"] if X.get("RM2_SUPP_DMPR_0") is not None else 0},
                    {'id': "1125084774763855880",
                     'angle': X["RM3_SUPP_DMPR_0"] if X.get("RM3_SUPP_DMPR_0") is not None else 0},
                    {'id': "1125084774763855882",
                     'angle': X["RM4_SUPP_DMPR_0"] if X.get("RM4_SUPP_DMPR_0") is not None else 0},
                    {'id': "1125084774763855881",
                     'angle': X["RM5_SUPP_DMPR_0"] if X.get("RM5_SUPP_DMPR_0") is not None else 0},
                    {'id': "1125084774763855879",
                     'angle': X["RM6_SUPP_DMPR_0"] if X.get("RM6_SUPP_DMPR_0") is not None else 0},
                    {'id': "1125084774763855878",
                     'angle': X["RM6_SUPP_DMPR_1"] if X.get("RM6_SUPP_DMPR_1") is not None else 0},
                    {'id': '1125084774763855885',
                     'angle': X["RM2_RET_DMPR_0"] if X.get("RM2_RET_DMPR_0") is not None else 0},
                    {'id': '1125084774763855888',
                     'angle': X["RM3_RET_DMPR_0"] if X.get("RM3_RET_DMPR_0") is not None else 0},
                    {'id': '1125084774763855887',
                     'angle': X["RM4_RET_DMPR_0"] if X.get("RM4_RET_DMPR_0") is not None else 0},
                    {'id': '1125084774763855886',
                     'angle': X["RM6_RET_DMPR_0"] if X.get("RM6_RET_DMPR_0") is not None else 0},
                    {'id': '1125084774763855873',
                     'angle': X["RM3_EXH_DMPR_0"] if X.get("RM3_EXH_DMPR_0") is not None else 0},
                    {'id': '1125084774763855874',
                     'angle': X["RM4_EXH_DMPR_0"] if X.get("RM4_EXH_DMPR_0") is not None else 0},
                    {'id': '1125084774763855872',
                     'angle': X["RM5_EXH_DMPR_0"] if X.get("RM5_EXH_DMPR_0") is not None else 0},
                    {'id': '1125084774763855875',
                     'angle': X["RM5_EXH_DMPR_1"] if X.get("RM5_EXH_DMPR_1") is not None else 0}
                ])

            # 设置风机
            if any([key in X for key in self.fan_keys]):
                for key in X:
                    if key in self.fan_keys:
                        if not (0 < X[key] <= 50):
                            raise ValueError(f"风机开度应在 (0, 50], but got {X[key]}")
                self.set_fan({
                    "MAU_FREQ": X["MAU_FREQ"] if X.get("MAU_FREQ") is not None else 30,
                    "AHU_FREQ": X["AHU_FREQ"] if X.get("AHU_FREQ") is not None else 30,
                    "EF_FREQ": X["EF_FREQ"] if X.get("EF_FREQ") is not None else 30
                })

            # 更新阻抗
            self.update_impedance(self.preset["requestSupplySystemId"])
            self.update_impedance(self.preset["requestReturnSystemId"])
            self.update_impedance(self.preset["requestExhaustSystemId"])
            # 调用 api 同步计算
            _result = self.compute_sync()
        except APIError as e:
            print(f"API 请求失败！{e}")
            return None
        except ValueError as e:
            print(f"参数错误！{e}")
            return None

        return _result

    def compute_supply(self, X) -> dict | None:
        """
        !!!!!!!!!!!!!! 该方法已废弃 !!!!!!!!!!!!!!
        从送风阀开度列表 --计算-> 实验室整体状态

        计算逻辑：
            1. 设置风阀开度
            2. 更新阻抗
            3. 同步计算

        Args:
            X (ndarray): 风阀开度列表
                X 的元素需要按照以下顺序（DamperProcessor 的 order 属性）排列：
                x[0]: '一更-SFF-01',
                x[1]: '二更-SFF-01',
                x[2]: '测试间一-SFF-01',
                x[3]: '测试间二-SFF-01',
                x[4]: '测试间三-SFF-01',
                x[5]: '洁净走廊-SFF-01',
                x[6]: '洁净走廊-SFF-02'

        Returns:
            _result (dict): 计算结果
        """
        # 检查：X 的元素需要在 0~90 之间
        if not all(0 <= x <= 90 for x in X):
            print(f"风阀开度：{X}")
            raise ValueError("风阀开度不在 0~90 之间！")
        try:
            # 调用 api 设置风阀角度
            self.set_dampers([
                {'id': "1095892728878727175", 'angle': X[0]},
                {'id': "1095892728878727169", 'angle': X[1]},
                {'id': "1095892728878727172", 'angle': X[2]},
                {'id': "1095892728878727174", 'angle': X[3]},
                {'id': "1095892728878727173", 'angle': X[4]},
                {'id': "1095892728878727170", 'angle': X[5]},
                {'id': "1095892728878727171", 'angle': X[6]},
            ])
            self.update_impedance(self.preset["requestSupplySystemId"])  # 调用 api 更新阻抗
            _result = self.compute_sync()  # 调用 api 同步计算
        except APIError as e:
            print(f"API 请求失败！{e}")
            return None

        return _result

    def get_system_ids(self, need_fresh_air=False):
        """随着项目 id、模型改变，系统 id 也会改变，需要重新获取

        Args:
            need_fresh_air (bool, optional): 是否需要新风系统。Defaults to False.
        """
        url = self.preset["url_prefix"] + f"/v2/projects/{self.preset['projectId']}/systems"
        payload = {}
        headers = {
            'Authorization': f'{self.auth_token}',
            'User-Agent': 'Apifox/1.0.0 (https://www.apifox.cn)'
        }

        rq = self.request_securely(method="GET",
                                   url=url,
                                   headers=headers,
                                   data=payload)
        if (rq is None) or (rq.status_code != 200):
            try:
                # 两种情况：1. 请求失败；2. 请求成功但是获取 token 失败
                raise APIError(f"Failed to get system ids, check your token.")
            except Exception as e:
                print(f'进程退出，错误信息：{e}')
                sys.exit(1)

        for sysitem in json.loads(rq.text):
            if sysitem["systemType"] == "system_supply":
                # 若 name 含有新风
                if "新风" in sysitem["name"] and need_fresh_air:
                    self.preset["requestSupplySystemId"] = sysitem["id"]
                # 若 name 不含新风
                if "新风" not in sysitem["name"]:
                    self.preset["requestSupplySystemId"] = sysitem["id"]
            elif sysitem["systemType"] == "system_return":
                self.preset["requestReturnSystemId"] = sysitem["id"]
            elif sysitem["systemType"] == "system_exhaust":
                self.preset["requestExhaustSystemId"] = sysitem["id"]

    def get_dampers(self):
        """todo: 存入 sqlite 数据库，然后风阀调用从数据库读取数据

        从同步计算结果中获取风阀列表，字段：
            roomNameToSupplyDamperConfigs
            roomNameToReturnDamperConfigs
            roomNameToExhaustDamperConfigs
        """
        computed_result = self.compute_sync()  # todo: 这里是第一次调用 api，可以存档作为缓存
        dampers = {}
        # 送风阀
        dampers.update(DataProcessor.dampers_result_to_dict(
            computed_result["roomNameToSupplyDamperConfigs"], "supply"))
        # 回风阀
        dampers.update(DataProcessor.dampers_result_to_dict(
            computed_result["roomNameToReturnDamperConfigs"], "return"))
        # 排风阀
        dampers.update(DataProcessor.dampers_result_to_dict(
            computed_result["roomNameToExhaustDamperConfigs"], "exhaust"))

        # json.dump(dampers, open("../ref/dampers.json", "w"), indent=4, ensure_ascii=False)
        self.dampers = dampers


if __name__ == "__main__":
    """示例代码
        更多用法参考 test/test_DampersProcessor_run_once.py
    """
    dp = DampersProcessor()

    """设定风阀配置"""
    dp.print_damper_configs()  # 打印风阀配置
    print("---------------------")

    """
    dict 的结构如下：
    {
        damper_uid:{
            "id": damper_id,
            "angle": damper_angle,
            "room": room_name,
            "systemT": "supply"/"return"/"exhaust",
            "name": None,
            "elementId": damper_elementId
            }
    }
    """

    new_damper_config = [
        {'id': "1095892728878727175", 'angle': 1.0},
        {'id': "1095892728878727169", 'angle': 2.0},
        {'id': "1095892728878727172", 'angle': 3.0},
        {'id': "1095892728878727174", 'angle': 4.0},
        {'id': "1095892728878727173", 'angle': 5.0},
        {'id': "1095892728878727170", 'angle': 6.0},
        {'id': "1095892728878727171", 'angle': 7.0},
    ]
    dp.set_dampers(new_damper_config)
    dp.print_damper_configs()
    print("---------------------")

    """更新阻抗"""
    dp.update_impedance(dp.preset["requestSupplySystemId"])

    """计算"""
    result = dp.compute_sync()
    now_time = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    json.dump(result, open(f"result_{now_time}.json", "w"))  # 保存结果

    """处理结果"""
    real_pressures = result["roomNameToRealPressure"]
    design_pressures = result["roomNameToDesignPressure"]
    print(f"实际压差：{real_pressures},\n设计压差：{design_pressures}")
