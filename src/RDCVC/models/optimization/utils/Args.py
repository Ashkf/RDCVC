from datetime import datetime
from enum import Enum
import os
from typing import Optional, Union
import toml
from dataclasses import asdict, dataclass, replace
from argparse import ArgumentParser
from geatpy import moea_NSGA2_templet

from CoreGA.problems.soea_nn_ALL_FreqPres_Pres import SN1

""" 
本文件包含了：
    1. problem 和 algorithm 的选取入口
    2. 参数系统


参数系统思路参考：https://zhuanlan.zhihu.com/p/568027631
结合 https://github.com/mivade/argparse_dataclass
"""

# -------------------------------------------------------- #
#         Optional mapping of problem and algorithm        #
# -------------------------------------------------------- #
problems_map = {
    "soea_nn": SN1,
    "soea_nn_ALL_FreqPres_Pres": SN1,
}

algorithms_map = {
    "nsga2": moea_NSGA2_templet,
}
# -------------------------------------------------------- #


@dataclass
class Arguments:
    """Arguments 类是一个用于储存参数的数据类。

    主要功能是储存参数，方便以属性的形式调用，减少后续代码中的魔法数字。
    """

    problem: str  # 问题类型
    algorithm: str  # 算法类型
    nind: int  # 种群规模
    maxgen: int  # 最大进化代数
    maxtime: int  # 最大运行时间 (s)
    nnpath: str  # 神经网络模型存档路径

    # --------------------- auto generate -------------------- #
    save_dir: str = ""  # 存档目录路径

    def check_args(self):
        # ------------------------ 参数验证逻辑 ------------------------ #
        if self.nind <= 0:
            raise ValueError("nind must be greater than 0")
        if self.maxgen <= 0:
            raise ValueError("maxgen must be greater than 0")
        if self.nnpath is not None:
            print("[WARN] 未指定神经网络模型。")

    def __post_init__(self):
        # 在初始化完成后执行的操作
        self.check_args()

    def update(self, new_args: Union["Arguments", dict]):
        """更新参数

        接收一个 Arguments 实例或一个 dict 以更新参数。
        更新后自动调用 check。

        Args:
            new_args (Union[Arguments, dict]): 新的参数
        """
        if isinstance(new_args, Arguments):
            updated_args = replace(self, **new_args.__dict__)
        elif isinstance(new_args, dict):
            updated_args = replace(self, **new_args)
        else:
            raise TypeError(
                "Arguments 的 update 方法仅支持传入 Arguments 或 dict 类型参数。"
            )
        # replace 创建一个新的 Arguments 并调用 __post_init__ 然后返回新的实例
        # 所以需要将更新后的属性赋值给当前实例
        self.__dict__.update(updated_args.__dict__)


class ArgsManagerMode(Enum):
    DEFAULT = 1
    CLI_FIRST = 2  # 命令行优先
    FILE_FIRST = 3  # 文件优先


class GAArgsManager:
    """GAArgsManager 类是一个用于管理 GA 参数的工具。它提供了以下主要功能：

        1. 接收和解析命令行输入。
        2. 检查参数之间的逻辑关系。
        3. 储存合法的参数列表。
        4. 参数保存为本地文件。
        5. 从本地文件读取参数。

    ArgsManager 对象可以传入代码，方便随时调用。
    """

    def __init__(self, mode: ArgsManagerMode, file_path: str = "./GAargs.toml"):
        """初始化 ArgsManager 对象

        Args:
            mode (ArgsManagerMode): 参数管理模式，用于决定参数来源优先级
            file_path (str): 保存和读取参数的文件路径
        """
        self.args: Optional[Arguments] = None
        self.mode = mode
        self.file_path = file_path

    def run(self):
        # ----------------------- 不同来源获取参数 ----------------------- #
        # args type: dict
        _args_f = self._load_file(self.file_path)
        _args_c = self._parse_args()

        # ----------------------- 合并不同来源参数 ----------------------- #
        # args type: Arguments
        if self.mode == ArgsManagerMode.CLI_FIRST or ArgsManagerMode.DEFAULT:
            _args = self.merge_arguments(_args_c, _args_f)
        elif self.mode == ArgsManagerMode.FILE_FIRST:
            _args = self.merge_arguments(_args_f, _args_c)

        # -------------------------- 后处理 ------------------------- #
        _args = self._process(_args)  # 处理
        self._store(_args, _args.save_dir)  # 保存

    @staticmethod
    def merge_arguments(
        argsA: Union[Arguments, dict], argsB: Union[Arguments, dict]
    ) -> Arguments:
        """这个方法用于合并两个参数实例。

        它首先检查两个实例中是否有相同的属性，然后将它们的值合并到新的实例中。

        Args:
            argsA (Union[Arguments, dict]): 高优先级 args
            argsB (Union[Arguments, dict]): 低优先级 args

        Returns:
            Arguments
        """
        if isinstance(argsA, Arguments) and isinstance(argsB, Arguments):
            _merged_args = Arguments()
            for attr in argsA.__annotations__:
                value = getattr(argsA, attr) or getattr(argsB, attr)
                setattr(_merged_args, attr, value)
        elif isinstance(argsA, dict) and isinstance(argsB, dict):
            _merged_args = argsB.copy()
            _merged_args.update(argsA)
            _merged_args = Arguments(**_merged_args)
        else:
            raise ValueError("Unsupported argument types.")

        return _merged_args

    def _parse_args(self) -> dict:
        """从命令行获取参数

        Returns:
            dict
        """
        _parser = self._set_args(ArgumentParser())
        _args = _parser.parse_args()
        return vars(_args)

    def _load_file(self, file_path) -> dict:
        """从本地文件读取参数

        Args:
            file_path (str): 配置文件地址

        Raises:
            TypeError: 文件类型错误
            ValueError: 文件格式错误

        Returns:
            dict
        """
        if not file_path.lower().endswith(".toml"):
            raise TypeError("配置文件类型错误，应为 toml 格式。")

        with open(file_path, "r") as f:
            try:
                _args = toml.load(f)
            except toml.TomlDecodeError as e:
                raise ValueError("文件不是合法的 TOML 格式。") from e

            return _args

    def _process(self, args):
        """参数后处理"""
        args.save_dir = _generate_archive_directory(args)
        return args

    def _store(self, args, file_dir):
        """参数保存为本地文件

        Args:
            file_dir (str): 存档目录
        """
        if args is None:
            raise ValueError("参数列表为空，无法保存。")

        file_path = os.path.join(file_dir, "GAargs.toml")
        with open(file_path, "w") as f:
            f.write("# This is a TOML document for GA\n")
            toml.dump(asdict(args), f)  # 储存参数为 toml 格式

    def _set_args(self, parser: ArgumentParser) -> ArgumentParser:
        """设置参数"""
        parser.add_argument(
            "--problem",
            type=str,
            choices=problems_map.keys(),
            required=True,
            help="问题类型",
        )
        parser.add_argument(
            "--algorithm",
            type=str,
            choices=algorithms_map.keys(),
            required=True,
            help="算法类型",
        )
        parser.add_argument(
            "-n", "--nind", type=int, default=100, required=True, help="种群规模"
        )
        parser.add_argument(
            "-g", "--maxgen", type=int, default=500, required=True, help="最大进化代数"
        )
        parser.add_argument(
            "-t",
            "--maxtime",
            type=int,
            default=999,
            help="最大运行时间 (s)",
        )
        parser.add_argument("--nnpath", type=str, help="神经网络模型存档路径")
        return parser


def _generate_archive_directory(args):
    """生成 GA 运行存档目录

    Returns:
        str: 存档目录
    """
    _base_directory = os.path.join(os.getcwd(), "checkpoints/GA")

    # ------------------------- 存档文件名 ------------------------ #
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%dT%H%M%S")
    dir_name = f"{args.problem}_{timestamp}"

    # ------------------------ 构建存档目录 ------------------------ #
    archive_directory = os.path.join(_base_directory, dir_name).replace("\\", "/")
    os.makedirs(archive_directory, exist_ok=False)

    return archive_directory


if __name__ == "__main__":
    # Example usage:
    args = {
        "nind": 100,
        "maxgen": 100,
        "maxtime": 60,
        "nnpath": "path/to/nn.h5",
        "problem": "tsp",
        "algorithm": "ga",
        "save_dir": "path/to/save",
    }

    arguments = Arguments()
    arguments.update(args)
    print(arguments.nind)  # Output: 100
    print(arguments.save_dir)  # Output: path/to/save
