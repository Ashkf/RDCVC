"""
基于 tensorboard 和 logging 的存图存曲线的 logger 类
"""

import logging
import os
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from . import myutils

"""
logging 模块的引入
1. 需要同时在终端和文件中进行写入 log 的能力；
2. 希望能够自定义 log 文档的格式，最好是能够加入时间这类信息；
    （这一点特别说明一下时间信息。一方面，方便我们能够通过时间快速的找到某次的训练内容；
    另一方面，还能够方便我们计算跑一个 epoch 所用时间这类小任务）；
3. 对日志信息进行级别的控制，如某些信息并不是必要的，
    只在特殊情况下才需要打印（如一些用于 debug 的信息），
    我们需要一个“开关”进行方便的指定。
"""


class Recoder:
    """一个数据统计工具

    在循环里 record 每次迭代的数据（比如各种评价指标 metrics），
    在每个 epoch 训练完成后，调用 summary，得到之前统计的指标各自的均值。

    这个工具在训练时嵌入到 Logger 中使用，
    在测试时由于不需要调用 tensorboard，所以直接被 evaluator 调用。
    """

    def __init__(self):
        """
        metrics: 每个 batch 的 metrics 的值，生命周期为一个 epoch
        metrics_buffer: 每个 epoch 的 metrics，生命周期为全部训练过程
        loss_weight: 每个 batch 的 loss_weight 的值，生命周期为一个 epoch
        (train/val)loss_buffer: 每 epoch 的 loss，生命周期为全部训练过程，
                                shape=[num_epochs,num_tasks]
        """
        self.metrics = {}
        self.metrics_buffer: dict[str:list] = {}
        self.loss_weight_buffer: Optional[np.ndarray] = None
        self.train_loss_buffer: Optional[np.ndarray] = None
        self.val_loss_buffer: Optional[np.ndarray] = None

    def record_metrics(self, name, value):
        """记录每个 metrics 的值"""
        if name in self.metrics.keys():
            self.metrics[name].append(value)
        else:
            # 新的指标
            self.metrics[name] = [value]
            self.metrics_buffer[name] = []

    def record_loss_weight(self, value):
        """记录 loss_weight 的值"""
        self.loss_weight_buffer = (
            np.concatenate((self.loss_weight_buffer, value), axis=0)
            if self.loss_weight_buffer is not None
            else value
        )

    def summary(self):
        self.summary_scaler()

    def summary_scaler(self):
        """求 self.metrics 里，各指标的均值"""
        # ----------------- metrics -----------------
        for key in self.metrics.keys():
            _mean = sum(self.metrics[key]) / len(self.metrics[key])  # 求均值
            self.metrics_buffer[key].append(_mean)  # 记录每个 epoch 的 metrics
        self.metrics = {}  # 清空 metrics
        # ----------------- loss-----------------
        train_loss = [
            v[-1].detach().cpu().numpy()
            for k, v in self.metrics_buffer.items()
            if k.split("/")[0] == "train" and k.split("/")[1].split("_")[0] == "loss"
        ]  # Take all the losses used as records to the cpu, shape=[num_tasks].
        train_loss = np.array(train_loss).reshape(1, -1)  # 转换为 numpy 数组
        self.train_loss_buffer = (
            np.concatenate((self.train_loss_buffer, train_loss), axis=0)
            if self.train_loss_buffer is not None
            else train_loss
        )

        val_loss = [
            v[-1].detach().cpu().numpy()
            for k, v in self.metrics_buffer.items()
            if k.split("/")[0] == "val" and k.split("/")[1].split("_")[0] == "loss"
        ]  # Take all the losses used as records to the cpu, shape=[num_tasks].
        val_loss = np.array(val_loss).reshape(1, -1)
        self.val_loss_buffer = (
            np.concatenate((self.val_loss_buffer, val_loss), axis=0)
            if self.val_loss_buffer is not None
            else val_loss
        )


class Checkpoint:
    """
    保存模型参数，以便继续训练
    """

    def __init__(self, model, optimizer, epoch, step):
        self.checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        }

    def save(self, model_dir):
        """保存训练状态

        保存到 model_dir/ckps/ckp_{epoch:04d}-{step:04d}.pth

        Args:
            model_dir: 模型保存的目录
        """
        ckps_dir = os.path.join(
            model_dir, "ckps"
        )  # ckps 是 checkpoints 的缩写，path=ckps/是保存模型参数的目录
        if not os.path.exists(ckps_dir):
            os.mkdir(ckps_dir)
        file_name = (
            f'ckp_E{self.checkpoint["epoch"]:04d}-B{self.checkpoint["step"]:04d}.pth'
        )
        path = os.path.join(ckps_dir, file_name)  # checkpoint 的路径

        # 保存 model、optimizer 的参数，以便继续训练
        torch.save(self.checkpoint, path)  # 保存模型参数，减小硬盘空间占用


class Logger:
    """
    将 tensorboard 的 SummaryWritter 包了一层，包含 recorder，还有 SummaryWritter；
    在训练或验证的每个 step 以 name-value 的形式 record 一下对应的曲线数据，
    name 最好用 train/xxx，val/xxx 这种形式，这样训练和测试的曲线会显示在两个图中，
    在每个 epoch 的最后一个 step 在每次训练或验证的 epoch 循环结束时，
        调用一次 save_curves 保存曲线；
    调用一次 save_checkpoint 保存模型参数；
    这些操作都在下面的 train.py 中体现。
    """

    def __init__(self, args):
        self.logger = self.load_logger(args)  # 加载 logger lib
        self.writer = self.create_SummaryWriter(args)
        self.recoder = Recoder()  # metric、loss 等数据的统计工具
        self.checkpoint = None
        self.log_args(args)  # 记录一下 args

    def log_args(self, args):
        """记录 args 参数"""
        for k in list(vars(args).keys()):
            self.logger.info(f"[args]=> {k}: {vars(args)[k]}")

    def create_SummaryWriter(self, args):
        """创建一个 SummaryWriter 对象

        当传入 model_dir 参数时，
            会在 model_dir 目录下生成一个 events.out.tfevents 文件，
        当传入 result_dir 参数时，
            会在 result_dir 目录下生成一个 events.out.tfevents 文件和 result.txt 文件。

        Args:
            args:

        Returns:
            writer: SummaryWriter 对象
        """
        if "model_dir" in dir(args):
            writer = SummaryWriter(args.model_dir)
        elif "result_dir" in dir(args):
            writer = SummaryWriter(args.result_dir)
        else:
            writer = SummaryWriter()
            self.logger.warning("Warning: no model_dir or result_dir is given.")
        return writer

    @staticmethod
    def tensor2img(tensor):
        # implement according to your data, for example call viz.py
        return tensor.cpu().numpy()

    def record_scalar(self, name, value):
        """完全等价于 Recoder..record()"""
        self.recoder.record_metrics(name, value)

    def record_loss_weight(self, value):
        """完全等价于 Recoder.record_loss_weight()"""
        self.recoder.record_loss_weight(value)

    def save_curves(self, epoch: int):
        # metrics
        for key, value in self.recoder.metrics_buffer.items():
            self.writer.add_scalar(key, value[-1], epoch)

        # loss_weight
        if not np.all(self.recoder.loss_weight_buffer == 1):
            # 将所有在 loss_weight_buffer 中的 loss_weight 都记录下来
            # loss_weight_buffer 的 shape 为 (epoch, num_tasks)
            # 每行是一个 epoch 中的 loss_weight
            loss_weights = {
                f"loss_weight_{i}": self.recoder.loss_weight_buffer[-1, i]
                for i in range(self.recoder.loss_weight_buffer.shape[1])
            }
            self.writer.add_scalars("loss_weights", loss_weights, epoch)

    def save_histogram(self, epoch: int):
        """保存模型参数的分布"""
        if not np.all(self.recoder.loss_weight_buffer == 1):
            loss_weight = self.recoder.loss_weight_buffer[-1, :]
            self.writer.add_histogram("loss_weight", loss_weight, epoch)

    def save_imgs(self, names2imgs, epoch):
        for name in names2imgs.keys():
            self.writer.add_image(name, self.tensor2img(names2imgs[name]), epoch)

    def save_last(self, item, epoch):
        """模仿 save_imgs，write 最后一个结果"""
        ...

    def save_checkpoint(self, model, optimizer, epoch, model_dir, step=0):
        """保存训练状态，包括模型参数、优化器参数、当前 epoch 和 step"""
        self.checkpoint = Checkpoint(model, optimizer, epoch, step)
        self.checkpoint.save(model_dir)  # save checkpoint to disk
        self.logger.debug(f"保存第 {epoch} 个 epoch 的 checkpoint")

    def save_final(self, model):
        """保存最终模型参数

        保存完整模型（不仅仅是模型参数）
        """
        filename = os.path.join(self.writer.log_dir, "final_model.pth")
        torch.save(model, filename)  # 保存模型参数，减小硬盘空间占用

    @staticmethod
    def load_logger(args):
        """加载 logger

        Args:
            args: 参数
        """

        mylogger = logging.getLogger(args.model_type)  # 创建一个 logger
        # mylogger.propagate = False  # 防止重复输出
        mylogger.setLevel(logging.DEBUG)  # 设置 log 的等级
        """设置控制台输出"""
        sHandler = logging.StreamHandler()  # 输出到屏幕的 handler
        # 设置输出格式
        formatter1 = logging.Formatter(
            fmt="[ %(asctime)s ] [%(levelname)s] %(message)s", datefmt="%d %H:%M:%S"
        )
        sHandler.setFormatter(formatter1)
        sHandler.setLevel(logging.INFO)  # 设置屏幕输出流的等级
        mylogger.addHandler(sHandler)  # 添加屏幕输出流的 handler
        """设置文件输出"""
        log_dir = None
        if "model_dir" in dir(args):
            log_dir = args.model_dir
        elif "result_dir" in dir(args):
            log_dir = args.result_dir
        # 输出到文件的 handler, 输出目录结构为：model_dir/log_2020-01-01T00-00-00.txt
        fHandler = logging.FileHandler(
            os.path.join(log_dir, "log" + "_" + myutils.get_now_time() + ".txt"),
            mode="w",
            encoding="utf-8",
        )
        formatter2 = logging.Formatter(
            fmt="[ %(asctime)s ] [%(levelname)s] %(message)s",
            datefmt="%a %b %d %H:%M:%S %Y",
        )
        fHandler.setFormatter(formatter2)
        fHandler.setLevel(logging.DEBUG)  # 设置文件输出流的等级
        mylogger.addHandler(fHandler)  # 添加文件输出流的 handler

        mylogger.info("mylogger is loaded")
        return mylogger
