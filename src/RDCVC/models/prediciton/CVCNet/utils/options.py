"""
这里定义了各种实验参数，其他模块多多少少都会与此有关，受此控制；

这里用命令行参数把各种参数通过某种方式传给程序。

命令行传参用到了 argparse 这个 lib

使用方法：
    1. 先用 parse_common_args 添加训练测试共用的一些参数
    2. 在 parse_train_args 和 parse_test_args 中调用这个公共的函数，
        这样可以避免有些参数在训练时写了，测试时忘了写，一跑就报错。
    3. parse_train_args 解析训练相关的参数，
    4. parse_test_args 解析测试相关的参数；

使用时，调用 prepare_train_args，就会创建一个包含所有公共参数和训练参数的 parser，
然后创建一个模型目录，并调用 save_args 函数保存所有参数，返回对应的 args。
保存参数这一步十分重要，能够避免模型训练完成之后忘记自己训练的模型配置这种尴尬局面。
测试时也类似，调用 prepare_test_args。
"""

import argparse
import os
import re

import yaml

from ..datasets.data_entry import type2data
from ..models.model_entry import type2help, type2model
from .myutils import get_now_time


def _set_common_args(parser):
    """设置 train/val 公用参数

    model_type: 模型的名字，配合 model 目录和 model_entry.py 使用；
    data_type：数据集的名字，配合 data 目录和 data_entry.py 使用；
    save_prefix：训练时：实验的名字，可以备注自己改了那些重要组件，具体的参数，会用于创建保存模型的目录；
                测试时：测试的名字，可以备注测试时做了哪些配置，会用于创建保存测试结果的目录；
    load_model_path：模型加载路径，训练时，作为预训练模型路径，测试时，作为待测模型路径，
                    有的人喜欢传入一个模型名字，再传入一个 epoch，但其实没啥必要，
                    就算要循环测多个目录我们也可以写 shell 生成对应的 load_model_path，
                    而且通常只需要测最后一个 epoch 的模型
    load_not_strict：我写了一个 load_match_dict 函数（utils/torch_utils.py），
                    允许加载的模型和当前模型的参数不完全匹配，可多可少，如果打开这个选项，就会调用此函数，
                    这样我们就可以修改模型的某个组件，然后用之前的模型来做预训练啦！
                    如果关闭，就会用 torch 原本加载的逻辑，要求比较严格的参数匹配；
    eval_path: 训练时可以传入验证集 path，测试时可以传入测试集 path；
    gpus：可以配置训练或测试时使用的显卡编号，在多卡训练时需要用到，
        测试时也可以指定显卡编号，绕开其他正在用的显卡，
        当然你也可以在命令行里 export CUDA_VISIBLE_DEVICES 这个环境变量来控制
    """

    parser.add_argument(
        "model_type",
        type=str,
        metavar="model_type",
        help=(
            f"【模型类型】可选：{list(type2model.keys())}\n"
            "================== model_type help ===================\n"
            f"{[f'{k}: {v}' for k, v in type2help.items()]}"
        ),
    )
    parser.add_argument(
        "data_type", type=str, help=f"[dataset type]：{list(type2data.keys())}"
    )
    parser.add_argument(
        "--save_prefix", type=str, default=".", help="Preserve the prefix of the model."
    )
    parser.add_argument(
        "--is_test", action="store_true", help="whether it is test mode"
    )
    parser.add_argument(
        "--load_model_path",
        type=str,
        default="",
        help="Load pre-trained model. Migration training is performed if provided.",
    )
    parser.add_argument(
        "--load_not_strict",
        action="store_true",
        help="Migration training allows loading only the generic state dictionary.",
    )
    parser.add_argument("--eval_path", type=str, default="", help="验证集文件路径")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="device: cuda or cpu. default: cuda",
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    parser.add_argument("--seed", type=int, default=None)
    return parser


def _set_train_args(parser):
    """设置 train 参数

    lr，momentum, beta, weight-decay: optmizer 相关参数，在 train.py 中初始化 optimizer
    model_dir：模型的存储目录，留空，不用传入，
                会在 get_train_model_dir 函数中确定这个字段的值，
                创建对应的目录，填充到 args 中，方便其他模块获得模型路径
    train_path：训练集 path
    batch_size：训练时的 batch size，有人可能会问，为啥测试时不用设置 batch size？
                主要是出于测试时的可视化需求
    epochs：模型训练 epoch 数

    """
    parser = _set_common_args(parser)  # 公用参数
    parser.add_argument(
        "--shuffle",
        type=bool,
        default="True",
        help="have the data reshuffled at every epoch",
    )
    parser.add_argument("--model_dir", type=str, help="!!! auto generated")
    parser.add_argument("--num_tasks", type=int, default=1, help="!!! auto generated")
    parser.add_argument("--num_batches", type=int, help="!!! auto generated")
    parser.add_argument(
        "--resume_path", type=str, default="", help="resume training from a checkpoint"
    )
    parser.add_argument("--train_path", type=str, default="/data/train.csv")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of total epochs to run"
    )
    parser.add_argument(
        "--print_freq_batch",
        default=1,
        type=int,
        help="Frequency of print in log file while training(batch)",
    )
    parser.add_argument(
        "--print_freq_epoch",
        default=1,
        type=int,
        help="Frequency of print in log file while training(epoch)",
    )
    parser.add_argument(
        "--save_every",
        default=10,
        type=int,
        help="Interval (epoch) for saving the model.",
    )
    parser.add_argument("--init_method", default="default", type=str)
    # --------------------- loss weight strategy ---------------------
    parser.add_argument(
        "--LWS",
        type=str,
        default="none",
        choices=["none", "DWA", "sum_loss"],
        help="loss_weight_strategy，指定多目标学习下 loss 优化策略。",
    )
    parser.add_argument(
        "--DWA_T",
        type=float,
        default=2,
        help="Temperature parameter for the DWA method, larger is smoother.",
    )
    parser.add_argument(
        "--DWA_limit",
        type=str,
        default="none",
        help="Limit the loss weight range, e.g. 0.1-10.",
    )
    parser.add_argument(
        "--DWA_weight",
        type=str,
        default="none",
        help="Specify the loss weight, e.g. 1,2,3,4,5,6,7,8,9,10.",
    )
    parser.add_argument(
        "--normalize_target",
        default="none",
        type=str,
        choices=["x", "y", "xy", "yx", "none"],
        help="归一化目标，可选：x，y，xy，none",
    )
    parser.add_argument(
        "--normalize_method",
        default="no_normalize_method",
        type=str,
        choices=["minmax", "zscore"],
        help="归一化方法，可选：minmax，zscore",
    )
    # --------------------- early stop ---------------------
    parser.add_argument("--earlystop", action="store_true", help="Early stop switch.")
    parser.add_argument(
        "--espatience", type=int, default=25, help="early stop patience"
    )
    parser.add_argument("--esdelta", type=float, default=0, help="early stop delta")
    # --------------------- optimizer ---------------------
    parser.add_argument("--lr", type=float, help="learning rate，学习率")
    parser.add_argument(
        "--momentum",
        type=float,
        metavar="M",
        help="momentum for sgd, alpha parameter for adam",
    )
    parser.add_argument(
        "--beta",
        type=float,
        metavar="M",
        help="beta parameters for adam，在 CoreNN/configs/config.yml 中存有默认值。",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        type=float,
        metavar="W",
        help="weight decay，在 CoreNN/configs/config.yml 中存有默认值。",
    )
    # --------------------- lr scheduler ---------------------
    parser.add_argument(
        "--lr_scheduler", action="store_true", help="LR scheduler switches"
    )
    parser.add_argument(
        "--lrspatience", type=int, default=20, help="LR scheduler patience"
    )
    parser.add_argument(
        "--lrsfactor", type=float, default=0.5, help="LR scheduler factor"
    )
    parser.add_argument(
        "--lrsmin", type=float, default=1e-6, help="LR scheduler min_lr"
    )
    return parser


def _set_test_args(parser):
    """解析 test 参数

    save_viz：控制是否保存可视化结果的开关，仅针对 cv 任务
    result_dir：可视化结果和测试结果的存储目录，留空，不用传入，
                会在 get_test_result_dir 中自动生成，自动创建目录，
                这个目录通常位于模型路径下，
                形如 checkpoints/model_name/checkpoint_num/val_info_save_prefix
    """
    parser = _set_common_args(parser)
    parser.add_argument(
        "--save_viz", action="store_true", help="save viz result in eval or not"
    )
    parser.add_argument(
        "--result_dir", type=str, default="", help="leave blank, auto generated"
    )
    parser.add_argument(
        "--from_checkpoint", action="store_true", help="test from a checkpoint"
    )
    return parser


def _build_train_model_dir(_args):
    """获取模型存档的 path

    生成模型训练存档，并将 path 存于 args 中，方便其他模块获取模型路径
    存档目录命名规则：
        <model_name>_<save_prefix>_BS<batch_size>_LR<learnnig_rate>_E<id>_<time>
        例如：
            mlp-L12_resume_DIYprefix_BS16_LR1e-3_E1_2020-12-12T12-12-12

    Returns:
        str: 模型目录
    """
    # 处理前缀
    prefix = _args.save_prefix
    if prefix == ".":
        # 若前缀为默认值
        prefix = (
            "_resume"
            if _args.resume_path != ""
            else "_pretrain" if _args.load_model_path != "" else ""
        )
    else:
        # 若前缀不为默认值
        prefix = (
            "_resume_"
            if _args.resume_path != ""
            else "_pretrain_" if _args.load_model_path != "" else "_"
        ) + prefix

    model_dir = _args.model_type  # 模型存档目录名：+ 模型名
    model_dir += f"{prefix}"  # 模型存档目录名：+ 保存前缀
    model_dir += f"_BS{_args.batch_size}"  # 模型存档目录名：+ batch size
    model_dir += f"_LR{_args.lr}"  # 模型存档目录名：+ 学习率
    model_dir += f"_EP{_args.epochs}"  # 模型存档目录名：+ epoch
    model_dir += f"_{get_now_time()}"  # 模型存档目录名：+ 时间

    model_path = os.path.join("checkpoints", "NN", model_dir)  # 模型存档路径
    if not os.path.exists(model_path):
        os.makedirs(model_path)  # 创建模型存档目录
    return model_path


def _build_test_result_dir(_args):
    """获取测试结果的存储目录

    将测试结果储存到 checkpoints/test_result/<model_name>_<time>/ 目录下

    Args:
        _args:测试参数
    """
    # 从 load_model_path 中获取模型的时间日期信息及 EPOCH、BATCH 信息
    matches_datetime = re.findall(
        r"\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}", _args.load_model_path
    )
    matches_epochAbatch = re.findall(r"E\d{4}-B\d{4}", _args.load_model_path)
    if len(matches_datetime) == 0:
        # 如果未找到时间日期信息
        datetime_info = "0000-00-00T00-00-00"
    elif len(matches_datetime) > 1:
        # 如果找到多个时间日期信息
        datetime_info = "9999-99-99T99-99-99"
    else:
        datetime_info = matches_datetime[0]
    if len(matches_epochAbatch) == 0:
        epochAbatch_info = "E0000-B0000"
    elif len(matches_epochAbatch) > 1:
        epochAbatch_info = "E9999-B9999"
    else:
        epochAbatch_info = matches_epochAbatch[0]

    now_time = get_now_time()  # 获取当前时间，格式为：2020-01-01T00-00-00
    # 生成结果目录，
    # 形如：checkpoints/test_result/2020-01-01T00-00-00_E0000-B0000_2020-01-01T00-00-00
    result_dir = os.path.join(
        "checkpoints",
        "NN_TestResult",
        datetime_info + "_" + epochAbatch_info + "_" + now_time,
    )
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)

    return result_dir


def _save_args(_args, save_dir):
    """保存 args"""
    args_path = os.path.join(save_dir, "args.yaml")

    with open(args_path, "w", encoding="UTF-8") as f:
        yaml.dump(vars(_args), f, allow_unicode=True)


def prepare_train_args():
    """准备训练参数"""
    # --------------------- get and check -------------------- #
    parser = argparse.ArgumentParser()
    _args = _set_train_args(parser).parse_args()
    _check_args(_args)

    # ------------------- process and save ------------------- #
    _args.model_dir = _build_train_model_dir(_args)

    # device
    _args.device = _process_device(_args.device, _args.gpus)

    _save_args(_args, _args.model_dir)

    return _args


def _process_device(device, gpus):
    """Process the device information.

    Args:
        device (str): The device to be used, either "cpu" or "cuda".
        gpus (int): The index of GPUs to be used.

    Returns:
        list: A list containing the device information.
    """
    if device == "cpu":
        device_info = ["cpu"]
    elif device == "cuda":
        device_info = ["cuda"] + [gpus]
    return device_info


def _check_args(_args):
    if _args.model_type.split("_")[0] not in type2model.keys():  # 检查 model_type 合法
        raise ValueError(f"模型类型 {_args.model_type} 不存在，请检查输入是否正确。")
    if _args.data_type not in type2data.keys():
        raise ValueError(f"数据集类型 {_args.data_type} 不存在，请检查输入是否正确。")
    if (
        _args.normalize_method == "no_normalize_method"
        and _args.normalize_target != "none"
    ):
        raise ValueError("未设定归一化方法，但设定了归一化目标，请检查输入是否正确。")
    if "mtl" in _args.model_type.split("_")[0]:
        # 获取 num_tasks
        # todo: 有待优化 num_tasks 的获取方法
        if "cvcnet" in _args.model_type.split("_")[0]:
            _args.num_tasks = 2
    if (
        _args.normalize_method != "no_normalize_method"
        and _args.normalize_target == "none"
    ):
        raise ValueError("未设定归一化目标，但设定了归一化方法，请检查输入是否正确。")
    if _args.resume_path != "" and _args.load_model_path != "":
        raise ValueError("resume_path 和 load_model_path 参数不能同时存在。")
    if _args.LWS != "none" and "mtl" not in _args.model_type.split("_")[0]:
        raise ValueError("LWS（loss_weight_strategy）只能在多目标学习场景下使用。")
    if _args.DWA_limit != "none":
        if len(_args.DWA_limit.split("-")) != 2:
            raise ValueError("DWA_limit 格式错误，应如：0.1-10。")
        _lower, _upper = _args.DWA_limit.split("-")
        if float(_lower) >= float(_upper):
            raise ValueError("DWA_limit 格式错误，左值应小于右值。")
        _args.DWA_limit = [float(_lower), float(_upper)]
    else:
        _args.DWA_limit = None
    if _args.DWA_weight != "none":
        # 解析为 list, float，要求为非零正值
        _args.DWA_weight = [float(i) for i in _args.DWA_weight.split(",")]
        if min(_args.DWA_weight) <= 0:
            raise ValueError("DWA_weight 格式错误，应为非零正值。")
        if len(_args.DWA_weight) != _args.num_tasks:
            raise ValueError(
                f"DWA_weight 格式错误，应为：{_args.num_tasks} 个逗号分隔的数字，"
                "例如：1,2,3,4,5,6,7,8,9,10。"
            )
    if _args.model_dir is not None or _args.num_batches is not None:
        raise ValueError("model_dir/num_batches 自动生成，不可手动指定。")
    if _args.espatience < 1:
        raise ValueError("early stop patience 必须 > 1。")
    if _args.esdelta < 0:
        raise ValueError("early stop delta 必须 >= 0。")
    if _args.lrspatience < 1:
        raise ValueError("lr_scheduler patience 必须 > 1。")
    if _args.lrspatience >= _args.espatience:
        raise ValueError("lr_scheduler patience 必须 < early stop patience。")
    if _args.lrsfactor <= 0:
        raise ValueError("lr_scheduler factor 必须 > 0。")
    if _args.lrsmin <= 0:
        raise ValueError("lr_scheduler min_lr 必须 > 0。")
    if _args.device == "cpu" and len(_args.gpus) > 1:
        raise ValueError("CPU 模式下，不可使用多卡。")
    if _args.device == "cuda" and len(_args.gpus) == 0:
        raise ValueError("CUDA 模式下，必须使用至少一块显卡。")


def prepare_test_args():
    """准备测试参数"""
    parser = argparse.ArgumentParser()  # 实例化 ArgumentParser 对象
    _args = _set_test_args(parser).parse_args()  # 设置并解析测试 args
    _check_args(_args)

    _args.result_dir = _build_test_result_dir(_args)  # 获取测试结果 path
    _save_args(_args, _args.result_dir)  # 保存测试 args
    return _args


if __name__ == "__main__":
    args = prepare_train_args()
    print(args)
