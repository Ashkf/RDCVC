"""
把测试（test）/验证（validation）的过程包成了一个 Evaluator。
对于测试和验证，实际上是一样的，都是对模型进行评估，只是数据集不同。
因此，我们把测试和验证的过程统称为评估（evaluate）。

训练（train）过程中，一般在每个 epoch 的最后会进行一次验证。
通过对验证结果的观察，可以判断模型是否过拟合，是否需要调整学习率等。
验证的目的是指导训练（调整模型参超参数），从而选择最优模型。
模型本身已经同时知道了输入和输出，所以从验证数据集上得出的误差（Error) 会有偏差（Bias)。

训练结束后，可能还需要对模型进行测试，测试的目的是评估模型的性能。


"""

import os

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .logger import Logger
from .model_executor import ModelExecutor


class Evaluator:
    def __init__(
        self,
        model,
        model_type,
        dataloader: DataLoader | None = None,
        logger: Logger | None = None,
        is_train=False,
        data: Tensor | None = None,
        metrics_computer=None,
    ):
        """初始化 evaluator

        Args:
            metrics_computer:
            model: 模型
            model_type: 模型类型
            dataloader: 数据加载器
            logger: 记录器
            is_train: 是否是训练模式。Evaluator 总是会调用 logger 记录 metrics，
                    但是在训练模式下，Evaluator 不会调用 logger 的 summary 方法。
            data: 直接传入数据，不使用 dataloader。
                如果传入了 data，那么 dataloader 将被忽略。并且 is_train 必须为 False。
        """

        self.model = model
        self.model_type = model_type
        self.dataloader = dataloader
        self.logger = logger
        self.is_train = is_train
        self.metrics_computer = metrics_computer
        self.data = data
        self.model_executor = ModelExecutor(model_type, model)

    def eval(self):
        """评估入口"""
        if not issubclass(self.model.__class__, torch.nn.Module):
            raise TypeError("model must be a subclass of torch.nn.Module")
        if self.data is not None:
            self.is_train = False
            ...

        self.model.eval()
        with torch.no_grad():
            for index_batch, item_batch in enumerate(self.dataloader):
                _data, _target = item_batch
                _pred = self.model(_data)
                _metrics = self.metrics_computer.comp_metrics(
                    _pred, _target, is_train=False
                )
                # ======================= record =======================
                for key in _metrics.keys():
                    self.logger.record_scalar(key, _metrics[key])
                # 记录最后一个 step 的结果
                # if index_batch == len(self.dataloader) - 1:
                #     self.logger.save_last()

            if not self.is_train:
                # 非训练模式下，调用 logger 的 summary 方法，得到所有的评估指标并输出
                self.logger.recoder.summary()
                _metrics = self.logger.recoder.metrics
                # write metrics to result dir,
                # you can also use pandas or other methods for better stats
                _f_name = os.path.join(self.logger.writer.log_dir, "result.txt")
                with open(_f_name, "w") as fd:
                    # 这边可以自定义写入的内容
                    fd.write(str(_metrics))  # 将 metrics 写入 result.txt
                    self.logger.logger.info(
                        f"Eval finished, results saved to {_f_name}"
                    )
