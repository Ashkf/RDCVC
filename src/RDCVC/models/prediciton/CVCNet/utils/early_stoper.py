"""
*
*
* File: early_stoper.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:03:55
* ----------------------------
* Modified: 2023-11-19 03:11:51
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from typing import Optional

import numpy as np


class EarlyStopper:
    """若验证集的损失在 patience 次迭代中没有改善，则提前停止训练。"""

    def __init__(self, logger, patience, delta):
        """
        Args:
            logger (Logger): 记录器，用于记录日志和保存模型
            patience (int): 验证损失没有改善的次数，超过该次数则提前停止训练。
            delta (float): 验证损失的最小变化，小于该值则认为验证损失没有改善。
        """
        self.logger = logger
        self.patience = patience
        self.delta = delta
        self.should_early_stop = False  # 是否提前停止训练
        self._counter = 0  # 记录验证损失没有改善的次数
        self._best_score = None  # 记录最佳验证损失
        self.val_loss_min = np.Inf  # 记录最佳验证损失的值

    def check(self, val_loss, loss_weight: Optional[np.ndarray] = None):
        _score = self._score(val_loss, loss_weight)  # 评分函数，验证损失越小，分数越高
        if self._best_score is None:  # 第一次迭代
            self._best_score = _score
        elif _score < self._best_score + self.delta:  # 验证损失没有改善
            self._counter += 1

            # 超过 patience/2 次没有改善时，记录日志
            if self._counter >= self.patience // 2:
                self.logger.logger.info(
                    f"EarlyStopping counter: {self._counter} / {self.patience}"
                )

            # 超过 patience 次没有改善时，提前停止训练
            if self._counter >= self.patience:
                self.logger.logger.info("Early Stopping...")
                self.logger.logger.info(f"Best valloss: {-self._best_score:.6f}")
                self.should_early_stop = True
        else:  # 验证损失改善
            self._best_score = _score
            self.val_loss_min = val_loss
            self._counter = 0

        return self.should_early_stop

    @staticmethod
    def _score(val_loss, loss_weight: Optional[np.ndarray] = None):
        """评分函数，验证损失越小，分数越高

        区分普通任务和 MTL。
            普通任务只有一个损失，直接取负为分数；
            MTL 有多个损失，取出权重，然后将损失乘以权重，再取负为分数。
        """
        if np.all(loss_weight == 1):
            _score = -np.mean(val_loss)  # 普通任务
        else:
            _score = -np.mean(val_loss)  # MTL

        return _score
