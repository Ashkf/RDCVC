import numpy as np
import torch


class LRScheduler:
    """
    学习率调度器。
    在 torch.optim.lr_scheduler.ReduceLROnPlateau 的基础上封装了一层。
    若验证集的损失 val_loss 在给定的 patience 次 epoch 迭代中没有改善，
    则学习率 lr 将按给定的 factor 减少。
    """

    def __init__(self, logger, optimizer, patience=5, min_lr=1e-6, factor=0.3):
        """
        new_lr = lr * factor

        Args:
            logger: 记录器，用于记录日志和保存模型
            optimizer: 调度器影响的优化器
            patience: 无提升等待的 epoch 数
            min_lr: 学习率更新后的最小值
            factor: 学习率的更新系数
        """
        self.logger = logger
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            patience=self.patience,
            factor=self.factor,
            min_lr=self.min_lr,
            verbose=True,
        )

    def check(self, val_loss, loss_weight: np.ndarray):
        if np.all(loss_weight == 1):
            val_loss = np.mean(val_loss)  # 普通任务
        else:
            val_loss = np.mean(val_loss)  # MTL

        self._lr_scheduler.step(val_loss)

        if self._lr_scheduler.num_bad_epochs > self.patience // 2:
            self.logger.logger.info(
                "LR_Scheduler counter: "
                + f"{self._lr_scheduler.num_bad_epochs} / {self.patience}",
            )
