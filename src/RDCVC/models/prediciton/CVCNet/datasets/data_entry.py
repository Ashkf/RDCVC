"""
提供了一个选择自定义的 dataloader 的入口

根据命令行参数，以字典的形式，快捷地选择要构造的 dataset。
如果你有更多的 dataset，可以继续扩展这部字典，字典访问是 O(1) 的，避免一堆 if-else 判断
有了 dataset，再用 pytorch 的 dataloader 接口包一下，可以支持 shuffle，多线程加载数据
"""

from torch.utils.data import DataLoader

from .cvcnet_dataset import CVCNetDataset
from .iotdp_dataset import IoTDamperDataset

type2data = {
    "cvcnet": CVCNetDataset,
    "iotdp": IoTDamperDataset,
}


def get_dataset(args, is_train=False, scaler=None, device=None):
    """构造了字典 type2data，返回 Dataset

    Args:
        scaler: 归一化器
        args (): 参数列表
        is_train (bool): 是否是训练集

    Returns:
        Dataset
    """
    return type2data[args.data_type](
        args, is_train=is_train, scaler=scaler, device=device
    )


def select_train_loader(args, logger, scaler=None, device=None) -> DataLoader:
    # 通常我们在 train 中需要 dataloader，在 eval/test 中需要 dataset。
    train_dataset = get_dataset(
        args, is_train=True, scaler=scaler, device=device
    )  # 获取 Dataset
    logger.info(f"{len(train_dataset)} samples found in train")
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=args.shuffle,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )


def select_eval_loader(args, logger, scaler=None, device=None) -> DataLoader:
    eval_dataset = get_dataset(args, scaler=scaler)
    logger.info(f"{len(eval_dataset)} samples found in eval")
    return DataLoader(
        eval_dataset,
        batch_size=len(eval_dataset),  # 读取全部测试集数据
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
