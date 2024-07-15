"""
提供 MetricsComputer 类，用于计算模型的各项指标

包括模型输出的各项指标的计算，loss/cost 的计算，以及 loss 的优化策略

指标 (metrics) 与 loss/cost 的区别：
    loss/cost 是模型优化的目标，是模型的损失函数，是模型的一部分
    metrics 是模型的评价标准，是模型的输出，是模型的输出的一部分
    loss/cost 从模型的角度出发，metrics 从用户的角度出发
    loss/cost 是面向算法的，metric 是面向任务的，本质计算方式一致

"""


import numpy as np
import torch
from torch import Tensor, mul
from torch.nn import Softmax
from torchmetrics.functional import (
    mean_squared_error,
    symmetric_mean_absolute_percentage_error,
)
from torchmetrics.functional.regression import (
    mean_absolute_error,
    mean_absolute_percentage_error,
)

from .scaler import ScalerMode


def ctfd(data: dict[str, Tensor], keys: list[str] | None = None) -> Tensor:
    """依据 keys 从 dict 中取出对应的 value，拼接成 tensor

    函数全称为：concat_tensor_from_dict

    Args:
        data(dict[str, Tensor]): 原始字典
        keys(list[str]): 需要拼接的 value 的 key 列表，默认为 None，即拼接所有 value
    """
    if keys is None:
        keys = list(data.keys())
    _sub_dict = {k: data[k] for k in keys}  # 根据 keys 从 dict 中取出对应的子字典
    _tensor = torch.stack([_sub_dict[k] for k in _sub_dict.keys()], dim=1)  # 子字典值拼接 tensor
    return _tensor


def dynamic_weight_average(loss_t_1, loss_t_2, T=2, limit_L=None, limit_R=None, c=None) -> Tensor:
    r"""DWA (Dynamic Weighting Average)

    Warnings:
        内部未对第一轮和第二轮的 loss 进行处理，
        默认 len(loss_t_1) == len(loss_t_2)

    动态平均权重，该情况下总体权重依然为：

    $$
    \mathcal{L}_{total}=\sum\nolimits_{k=1}^{K}\lambda_{k}\mathcal{L}_{k}
    $$

    式中 $K$ 为任务数， $\lambda_{k}$ 为第 $k$ 个任务的损失权重，其定义如下：

    $$
    \lambda_{k}(t):=
    \frac{K \exp \left(w_{k}(t-1) / T\right)}{\sum_{i} \exp\left(w_{i}(t-1) / T\right)}
     = K\text{softmax}(w_{k}(t-1) / T)
    $$

    式中 $T$ 为平滑权重的温度，越大则 task 的权重分布越均匀。
    $w_{k}(t-1)$ 为上一轮以及上上轮的 loss 比率，代表不同 task 的学习速率，定义如下：

    $$
    w_{k}(t-1)=\frac{\mathcal{L}_{k}(t-1)}{\mathcal{L}_{k}(t-2)}
    $$

    Args:
        loss_t_1: 第 t-1 轮的 loss
        loss_t_2: 第 t-2 轮的 loss
        T: 温度参数
        limit_low: lambda 下限 （optional）
        limit_high: lambda 上限 （optional）
        c: 额外添加的权重参数 （optional）
    Note:
        1. 该函数内部未对第一轮和第二轮的 loss 进行处理，默认 len(loss_t_1) == len(loss_t_2)
        2. 若上下限不为 None，则会对 lambda 进行限制
        3. 若 c 不为 None，则会对 lambda 进行额外的权重调整
        4. 若不启用上下限和额外权重调整，则为经典的 DWA
    """
    assert len(loss_t_1) == len(loss_t_2), "loss_t_1 and loss_t_2 must have the same length"

    K = len(loss_t_1)  # K: number of tasks

    w = Tensor(
        [(l_1 / l_2) for l_1, l_2 in zip(loss_t_1, loss_t_2, strict=True)]
    )  # w: loss ratio, quicker task has smaller loss ratio

    # c takes normalization
    c = Tensor(([1] * K) if c is None else c)
    c = c / torch.sum(c)

    _lambda = K * Softmax(dim=0)(mul(w, c) / T)

    if limit_L is not None and limit_R is not None:
        _lambda = torch.clamp(_lambda, limit_L, limit_R)  # limit lambda

    return _lambda


def uncertainty_to_weigh_losses():
    ...


class MetricsComputer:
    """Compute metrics for model."""

    def __init__(
        self,
        model_type: str,
        loss_weight_strategy: str | None = None,
        logger=None,
        scaler=None,
        num_tasks: int | None = None,
    ):
        """传入模型类型，根据模型类型调用对应的计算函数

        Args:
            scaler:
            model_type(str): 模型类型
            loss_weight_strategy(str): 损失优化策略
            logger: 记录器
            scaler: 归一化器
            num_tasks(int): 任务数
        """
        self.model_type = model_type
        self.loss_weight_strategy = loss_weight_strategy  # 损失优化策略
        self.logger = logger
        self.metrics = None  # 用于保存计算好的指标
        self.scaler = scaler
        self.num_tasks = num_tasks  # 任务数

    def comp_metrics(self, pred, target, is_train: bool) -> dict:
        _model_name = self.model_type.split("_")[0]
        match _model_name:
            case "cvcnet-mtl-mlp" | "split-mtl-mlp" | "split-mtl-kane":
                _metrics = self._comp_metrices_byTasksType(pred, target, is_train=is_train)
            case "dense-mtl":
                _metrics = self._comp_metrics_byTarget(pred, target, is_train=is_train)
            case "mlp" | "dapn12" | "kane":
                _metrics = self._comp_metrics_IoTDamper(pred, target, is_train=is_train)
            case _:
                raise ValueError(f"Unknown model type: {self.model_type}")
        self.metrics = _metrics
        return _metrics

    def comp_loss(self, metrics: dict | None = None) -> torch.Tensor:
        """从 metrics 提取/计算损失

        Args:
            metrics(dict): 模型的各项指标，默认为 None，即使用 self.metrics
        """
        _model_type = self.model_type
        # 获取模型类型的前缀，例如 'cpn-mlp_6_6'->'cpn-mlp'
        _model_prefix = _model_type.split("_")[0]
        if metrics is None:
            # Try to retrieve metrics from self.metrics, if self.metrics is None,
            # raise an exception
            try:
                metrics = self.metrics
            except AttributeError as err:
                raise AttributeError("self.metrics is None, please compute metrics first.") from err

        # ---------------------------------- 计算损失 ----------------------------------
        if "mtl" in _model_prefix:
            loss = [
                v for k, v in metrics.items() if k.split("/")[0] == "train" and k.split("/")[1].split("_")[0] == "loss"
            ]
            loss = self._use_loss_weight(loss)
        else:
            loss = metrics["train/loss"]
        return loss

    def calc_loss_weight(self, args):
        if self.loss_weight_strategy in ["none", "sum_loss"]:
            _weight = np.ones((1, self.num_tasks))
        elif self.loss_weight_strategy in ["DWA", "LDWA", "WDWA", "LWDWA"]:
            _loss_buffer = self.logger.recoder.train_loss_buffer
            _loss_weight_buffer = self.logger.recoder.loss_weight_buffer

            # ----------------- cal balancing weight ----------------- #
            if _loss_buffer is None:
                _weight = np.ones((1, self.num_tasks))  # 第一轮权重为 1
            elif len(_loss_buffer) == 1:
                _weight = np.ones_like(_loss_buffer[-1])  # 第二轮权重为 1
            else:
                _loss_t_1 = _loss_buffer[-1]  # 上轮的 loss
                _loss_t_2 = _loss_buffer[-2]  # 上上轮的 loss
                _weight = dynamic_weight_average(
                    _loss_t_1,
                    _loss_t_2,
                    T=args.DWA_T,
                    limit_L=args.DWA_limit[0] if args.DWA_limit is not None else None,
                    limit_R=args.DWA_limit[1] if args.DWA_limit is not None else None,
                    c=args.DWA_weight,
                )  # 计算权重
        else:
            raise NotImplementedError(f"Unknown loss optimization strategy: {self.loss_weight_strategy}")
        self.logger.record_loss_weight(_weight.reshape(1, -1))  # 写入 logger.recorder

    @staticmethod
    def _calc_mae(pred, ground):
        return mean_absolute_error(pred, ground)

    @staticmethod
    def _calc_mse(pred, ground, dim=None):
        squared_errors = torch.square(pred - ground)
        return torch.mean(squared_errors, dim=dim)

    @staticmethod
    def _calc_rmse(pred, ground):
        return mean_squared_error(pred, ground, squared=False)

    @staticmethod
    def _calc_mape(pred, ground):
        return mean_absolute_percentage_error(pred, ground)

    @staticmethod
    def _calc_smape(pred, ground):
        return symmetric_mean_absolute_percentage_error(pred, ground)

    def _calc_loss_l2(self, pred, ground):
        return self._calc_mse(pred, ground, dim=0)

    def _calc_loss_l1(self, pred, ground):
        return self._calc_mae(pred, ground)

    def _comp_metrics_IoTDamper(self, pred: Tensor, target, is_train) -> dict:
        """IoTDamper 风阀任务的指标计算


        Args:
            pred(Tensor): 模型输出。Tensor(batch_size, 1)
            target(Tensor): 目标值。Tensor(batch_size, 1)

        UPDATE: 2024-05-31
        """
        # ---------------------- 反归一化 (若对标签采取归一化) ----------------------
        _pred = self.scaler.scale(pred, "y", is_train, mode=ScalerMode.INVERSE_NORMALIZATION).to("cpu")
        _target = self.scaler.scale(target, "y", is_train, mode=ScalerMode.INVERSE_NORMALIZATION).to("cpu")

        # ------------------------ metrics ----------------------- #
        prefix = "train/" if is_train else "val/"  # 前缀，用于区分训练和验证
        return {
            f"{prefix}loss": self._calc_loss_l2(_pred, _target),
            f"{prefix}rmse": self._calc_rmse(_pred, _target),
            f"{prefix}mae": self._calc_mae(_pred, _target),
            f"{prefix}mape": self._calc_mape(_pred, _target),
        }

    def _comp_metrices_byTasksType(self, pred: list[Tensor], target, is_train) -> dict:
        """按照任务类别计算指标

        RDCVC 任务中，分为两类任务，一类是系统风量，一类是区域压差

        Args:
            pred(List[Tensor]): 模型输出.
                [Tensor(batch_size, D_t) * num_tasks], D_t 为任务 t 的输出维度
            target(Tensor): 目标值。Tensor(batch_size, D) D 为所有任务的输出维度之和

        UPDATE: 2023-12-03 20:00
        """
        # ------------------------- loss ------------------------- #
        # per loss shape: (num_feature, )
        assert pred[0].shape[1] == 4, "The task involving airflow has four specific targets."
        _loss_airflow = self._calc_loss_l2(pred[0], target[:, :4])
        assert pred[1].shape[1] == 6, "The task involving pressure has six specific targets."
        _loss_rm_pres = self._calc_loss_l2(pred[1], target[:, 4:])

        # ---------------------- 反归一化 (若对标签采取归一化) ----------------------
        # !!! 会破坏 tensor 的计算图
        _pred = torch.cat(pred, dim=1)  # Tensor(batch_size, num_target)
        _pred = self.scaler.scale(_pred, "y", is_train, mode=ScalerMode.INVERSE_NORMALIZATION).to("cpu")
        _target = self.scaler.scale(target, "y", is_train, mode=ScalerMode.INVERSE_NORMALIZATION).to("cpu")

        # ------------------------ metrics ----------------------- #
        prefix = "train/" if is_train else "val/"  # 前缀，用于区分训练和验证
        metrics = {
            prefix + "loss_airflow": torch.mean(_loss_airflow),
            prefix + "loss_rm_pres": torch.mean(_loss_rm_pres),
            prefix + "rmse_airflow": self._calc_rmse(_pred[:, :4], _target[:, :4]),
            prefix + "mae_airflow": self._calc_mae(_pred[:, :4], _target[:, :4]),
            prefix + "rmse_rm_pres": self._calc_rmse(_pred[:, 4:], _target[:, 4:]),
            prefix + "mae_rm_pres": self._calc_mae(_pred[:, 4:], _target[:, 4:]),
        }
        return metrics

    def _comp_metrics_byTarget(self, pred: tuple[Tensor] | list[Tensor], target, is_train):
        """按照每个目标计算指标"""

        # pred: Tuple(Tensor(batch_size, 1) * num_tasks)
        # target: Tensor(batch_size, num_task)

        # ------------------------- loss ------------------------- #
        pred = torch.cat(pred, dim=1)  # Tensor(batch_size, num_task)
        _losses = self._calc_loss_l2(pred, target)

        # ---------------------- 反归一化 (若对标签采取归一化) ----------------------
        _pred = self.scaler.scale(pred, "y", is_train, mode=ScalerMode.INVERSE_NORMALIZATION).to("cpu")
        _target = self.scaler.scale(target, "y", is_train, mode=ScalerMode.INVERSE_NORMALIZATION).to("cpu")

        # ------------------------ metrics ----------------------- #
        prefix = "train/" if is_train else "val/"  # 前缀，用于区分训练和验证
        return {
            prefix + "loss_tot_fv": _losses[0],
            prefix + "loss_tot_sv": _losses[1],
            prefix + "loss_tot_ev": _losses[2],
            prefix + "loss_tot_rv": _losses[3],
            prefix + "loss_rm_pres_1": _losses[4],
            prefix + "loss_rm_pres_2": _losses[5],
            prefix + "loss_rm_pres_3": _losses[6],
            prefix + "loss_rm_pres_4": _losses[7],
            prefix + "loss_rm_pres_5": _losses[8],
            prefix + "loss_rm_pres_6": _losses[9],
            prefix + "mape_tot_fv": self._calc_mape(_pred[:, 0], _target[:, 0]),
            prefix + "rmse_tot_fv": self._calc_rmse(_pred[:, 0], _target[:, 0]),
            prefix + "mape_tot_sv": self._calc_mape(_pred[:, 1], _target[:, 1]),
            prefix + "rmse_tot_sv": self._calc_rmse(_pred[:, 1], _target[:, 1]),
            prefix + "mape_tot_ev": self._calc_mape(_pred[:, 2], _target[:, 2]),
            prefix + "rmse_tot_ev": self._calc_rmse(_pred[:, 2], _target[:, 2]),
            prefix + "mape_tot_rv": self._calc_mape(_pred[:, 3], _target[:, 3]),
            prefix + "rmse_tot_rv": self._calc_rmse(_pred[:, 3], _target[:, 3]),
            prefix + "mape_rm_pres": self._calc_mape(_pred[:, 4:10], _target[:, 4:10]),
            prefix + "rmse_rm_pres": self._calc_rmse(_pred[:, 4:10], _target[:, 4:10]),
        }

    def _use_loss_weight(self, _loss: list[Tensor]) -> Tensor:
        """Use loss weight to compute loss."""
        assert all(_l.shape == () for _l in _loss), "Per loss must be a scalar."
        _weight = self.logger.recoder.loss_weight_buffer[-1, :]
        for _l, _w in zip(_loss, _weight, strict=True):
            _l *= _w
        _loss = sum(_loss)
        return _loss
