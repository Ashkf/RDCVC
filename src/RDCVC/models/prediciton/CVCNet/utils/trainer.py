"""
* trainer of CVCNet
*
* File: trainer.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:03:55
* ----------------------------
* Modified: 2024-02-06 23:09:16
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

# built-in
import os
import pickle
import random

import numpy as np

# third-party
import torch
import torch.optim
import torch.utils.data
import yaml
from tqdm import trange

# self-built
from ..datasets.data_entry import select_eval_loader, select_train_loader
from ..models import model_entry
from .early_stoper import EarlyStopper
from .evaluator import Evaluator
from .logger import Logger
from .metrics_computer import MetricsComputer
from .model_executor import ModelExecutor
from .options import prepare_train_args
from .scaler import Scaler
from .scheduler import LRScheduler
from .torch_utils import load_match_dict


class Trainer:
    """
    The Trainer class is responsible for training and evaluating a model.

    Attributes:
        args: The parsed training arguments.
        default_config: The default configuration for the trainer.
        logger: The logger for logging training progress.
        scaler: The feature scaler for preprocessing data.
        train_dataloader: The dataloader for training data.
        val_dataloader: The dataloader for validation data.
        model: The model to be trained.
        metrics_computer: The metrics computer for evaluating model performance.
        evaluator: The evaluator for evaluating model performance.
        optimizer: The optimizer for updating model parameters.
        lr_scheduler: The learning rate scheduler for adjusting learning rate during training.
        model_executor: The model executor for executing model forward pass.
        early_stopper: The early stopper for early stopping training based on validation loss.
    """

    def __init__(self):
        """
        Initializes the Trainer object.
        """
        self.args = prepare_train_args()
        self.set_random_seed(self.args.seed)
        self.default_config = self.prepare_default_config()
        self.logger = Logger(self.args)  # 初始化 Logger
        self.scaler = Scaler(self.args)  # 初始化特征缩放器
        self.train_dataloader = select_train_loader(
            self.args, self.logger.logger, self.scaler
        )
        self.val_dataloader = select_eval_loader(
            self.args, self.logger.logger, self.scaler
        )
        self.model = self.prepare_model()
        self.metrics_computer = self.prepare_metrics_computer()
        self.evaluator = self.prepare_evaluator()
        self.optimizer = self.prepare_optimizer()
        self.lr_scheduler = LRScheduler(
            self.logger,
            self.optimizer,
            patience=self.args.lrspatience,
            factor=self.args.lrsfactor,
            min_lr=self.args.lrsmin,
        )
        self.model_executor = ModelExecutor(self.args.model_type, self.model)
        self.early_stopper = EarlyStopper(
            self.logger, patience=self.args.espatience, delta=self.args.esdelta
        )

    # @littletimer
    def train(self):
        """训练入口

        迭代 epochs 次，每次调用 train_per_epoch, val_per_epoch 执行训练和验证，
        再调用 logger 存储曲线和图像。若提供了 resume_path，则从 checkpoint 恢复训练，
        否则从头开始训练。
        """

        # 检查是否提供了 resume_path。
        if self.args.resume_path != "":
            start_epoch = self.logger.checkpoint["epoch"] + 1  # read epoch
        else:
            start_epoch = 0  # 从头开始训练，start_epoch 为 0

        self.logger.logger.info("开始训练...")
        # todo:warmup
        for epoch in trange(
            start_epoch, self.args.epochs, desc="train", leave=False, colour="GREEN"
        ):
            self.train_one_epoch(epoch)
            self.evaluator.eval()
            self.logger.recoder.summary()
            self.logger.save_curves(epoch)  # 处理 metrics、保存曲线数据
            # self.logger.save_histogram(epoch)  # 保存直方图

            # ------------------save checkpoint------------------
            _flag_epoch = epoch % self.args.save_every == 0
            _flag_last = epoch == self.args.epochs - 1
            if _flag_epoch or _flag_last:
                self.logger.save_checkpoint(
                    self.model, self.optimizer, epoch, self.args.model_dir
                )
            # ------------------adjust lr------------------
            if self.args.lr_scheduler:
                self.lr_scheduler.check(
                    self.logger.recoder.val_loss_buffer[-1],
                    self.logger.recoder.loss_weight_buffer[-1],
                )
            # ------------------early stopping------------------
            if self.args.earlystop and self.early_stopper.check(
                self.logger.recoder.val_loss_buffer[-1],
                self.logger.recoder.loss_weight_buffer[-1],
            ):
                self.logger.save_checkpoint(
                    self.model, self.optimizer, epoch, self.args.model_dir
                )
                break
        self.logger.save_final(self.model)  # train 完成后保存最终完整的模型
        self.logger.logger.info("训练完成，保存最终模型")

    def train_one_epoch(self, index_epoch):
        """训练核心代码

        1. 切换模型到训练模式
        2. 遍历整个 train_loader
           1. 调用 step 函数进行数据拆包，再执行模型 forward，获取预测结果
           2. 调用 compute_metrics 计算 metrics
           3. 计算 loss
           4. 执行 反向传播
           5. 记录 每次迭代的 metrics。调用 logger 的 record 函数
           6. 记录 最后一个 step 结果。调用 logger 的 save_last 函数
           7. 监控 训练过程。根据 print_freq，每隔一段时间打印日志方便观察。

        Args:
            index_epoch (int): 训练代数
        """
        self.model.train()  # switch to train mode
        self.metrics_computer.calc_loss_weight(self.args)  # 计算 loss_weight
        _device = self.args.device[0]
        """遍历整个 train_loader """
        for index_batch, items_batch in enumerate(self.train_dataloader):
            _data, _target = items_batch
            self.optimizer.zero_grad()
            _pred = self.model(_data)
            _metrics = self.metrics_computer.comp_metrics(_pred, _target, is_train=True)
            _loss = self.metrics_computer.comp_loss()  # 计算 loss
            _loss.backward()  # 反向传播
            self.optimizer.step()  # 使用预先设置的优化器根据当前梯度对权重进行更新

            # =======================after backward=======================
            # logger  record
            for key in _metrics.keys():
                self.logger.record_scalar(key, _metrics[key])
            # 记录最后一个 step 的结果
            # if index_batch == len(self.train_loader) - 1:
            #     self.logger.save_last(...)
            # 监视训练过程
            _flag_batch = index_batch % self.args.print_freq_batch == 100  # 每 100batch
            _flag_epoch = index_epoch % self.args.print_freq_epoch == 0  # 每 1epoch
            if _flag_batch and _flag_epoch:
                self.logger.logger.debug(
                    f"Train: Epoch {index_epoch:04d} "
                    f"| batch {index_batch:04d}  | Loss {_loss}"
                )

    @staticmethod
    def set_random_seed(seed):
        """设置随机种子

        Args:
            seed (int): 随机种子
        """
        if seed is None:
            return
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    @staticmethod
    def gen_item(item, pred, label, is_train):
        # override this method according to your need
        # prefix = 'train/' if is_train else 'val/'
        # return {
        #     prefix + 'item': item[0],
        #     prefix + 'pred': pred[0],
        #     prefix + 'label': label[0]
        # }
        ...

    @staticmethod
    def gen_imgs_to_write(img, pred, label, is_train):
        # override this method according to your visualization
        prefix = "train/" if is_train else "val/"
        return {
            prefix + "img": img[0],
            prefix + "pred": pred[0],
            prefix + "label": label[0],
        }

    def prepare_model(self):
        """初始化模型"""
        model = model_entry.select_model(
            self.args.model_type
        )  # 选择模型结构（by --model_type）
        if self.args.resume_path != "":
            # 若为 resume 训练，则加载 checkpoint
            self.logger.logger.warning(
                f"Resume training from checkpoint: {self.args.resume_path}"
            )
            self.logger.checkpoint = torch.load(self.args.resume_path)
            model.load_state_dict(self.logger.checkpoint["model_state_dict"])
        else:
            if self.args.load_model_path != "":
                self.logger.logger.info("pretrain 任务")
                # 若为 pretrain 训练，则加载已有模型参数
                if self.args.load_not_strict:
                    self.logger.logger.info("不严格读取对应的模型参数，局部 pretrain")
                    # 不严格读取对应的模型参数，作为一种局部 pretrain
                    load_match_dict(model, self.args.load_model_path)
                else:
                    self.logger.logger.info("严格读取对应的模型参数，全局 pretrain")
                    _ = torch.load(self.args.load_model_path)  # 从文件中读取
                    # 严格读取模型参数，需要参数与模型结构完全对应
                    if isinstance(_, dict):
                        self.logger.logger.info("读取到模型参数字典，开始加载")
                        model.load_state_dict(_["model_state_dict"])
                    else:
                        self.logger.logger.info("读取到完整模型，开始加载")
                        model = _
                self.logger.logger.warning(
                    "Pretrained model parameters"
                    f" are being used: {self.args.load_model_path}"
                )
            else:
                # 若为新训练，则初始化模型参数
                self.logger.logger.info("新训练任务，初始化模型")
                # 初始化参数，init_model 内部未具体实现
                model = model_entry.init_model(model, self.args, self.logger.logger)

        model = model_entry.equip_device(model, self.args.device)

        # print(model)
        # _in_dim = model.backbone.layers[0].in_features
        # summary(model, input_size=(1, _in_dim))  # 打印模型结构
        # model_graph = draw_graph(
        #     model,
        #     input_size=(1, _in_dim),
        #     graph_name=self.args.model_type,
        #     roll=True,
        #     expand_nested=True,
        #     # save_graph=True,
        #     filename="model_graph",
        #     directory=self.args.model_dir,
        # )  # 绘制模型结构图
        # model_graph.visual_graph.node_attr["fontname"] = "Times-Roman"
        # model_graph.visual_graph.render(format="png")  # 保存模型结构图

        return model

    def prepare_optimizer(self):
        """初始化优化器

        优化器参数优先级：：
        1. 命令行参数指定：以命令行输入参数为准
        2. 恢复训练：从 checkpoint 中读取
        3. 未指定参数：从 config 文件中读取

        运行逻辑参考 ref/prepare_optimizer.drawio

        多步长 SGD 继续训练：https://www.cnblogs.com/devilmaycry812839668/p/10630302.html
        """
        # 优化器初始化
        # 从 config 文件中读取优化器参数默认值
        lr = self.default_config["optimizer"]["lr"]
        momentum = self.default_config["optimizer"]["momentum"]
        beta = self.default_config["optimizer"]["beta"]
        wt_decay = self.default_config["optimizer"]["weight_decay"]

        # 是否为 resume 模式，若是，则从 checkpoint 恢复优化器参数
        if self.args.resume_path != "":
            ckp_optim = self.logger.checkpoint["optimizer_state_dict"]
            # 从 checkpoint 恢复优化器参数
            lr = ckp_optim["param_groups"][0]["lr"]
            momentum = ckp_optim["param_groups"][0]["betas"][0]
            beta = ckp_optim["param_groups"][0]["betas"][1]
            wt_decay = ckp_optim["param_groups"][0]["weight_decay"]
            self.logger.logger.info("注意：从 checkpoint 恢复优化器参数")

        # 是否提供 lr 参数，若是，则以命令行输入参数为准
        lr = self.args.lr if self.args.lr is not None else lr
        # 是否提供 momentum 参数，若是，则以命令行输入参数为准
        momentum = self.args.momentum if self.args.momentum is not None else momentum
        # 是否提供 beta 参数，若是，则以命令行输入参数为准
        beta = self.args.beta if self.args.beta is not None else beta
        # 是否提供 weight_decay 参数，若是，则以命令行输入参数为准
        wt_decay = (
            self.args.weight_decay if self.args.weight_decay is not None else wt_decay
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr, betas=(momentum, beta), weight_decay=wt_decay
        )

        return optimizer

    def prepare_scaler(self, save_dir):
        """初始化特征缩放器

        Args:
            save_dir: 保存特征缩放器的目录

        Returns:
            scaler: 特征缩放器
        """
        # todo: 若为 resume 训练，则加载原有的 scaler
        # 检查目录下是否有 scalers.pkl
        file_path = os.path.join(save_dir, "scalers_dict.pkl")
        if os.path.isfile(file_path):
            self.logger.logger.info("找到已存在的 scalers。")
            with open(file_path, "rb") as f:
                scalers_dict = pickle.load(f)
            return scalers_dict
        else:
            self.logger.logger.warn("未找到已存在的 scalers。")
            return

    @staticmethod
    def prepare_default_config():
        """从 config.yml 中读取默认配置"""
        with open("src/RDCVC/models/prediciton/CVCNet/configs/config.yml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

    def prepare_evaluator(self):
        return Evaluator(
            self.model,
            self.args.model_type,
            self.val_dataloader,
            self.logger,
            is_train=True,
            metrics_computer=self.metrics_computer,
        )

    def prepare_metrics_computer(self):
        return MetricsComputer(
            self.args.model_type,
            self.args.LWS,
            self.logger,
            self.scaler,
            num_tasks=self.args.num_tasks,
        )
