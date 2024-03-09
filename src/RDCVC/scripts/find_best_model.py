"""
* 用于读取并检查 checkpoints 目录下的所有训练记录，并获取最佳的模型。
*
* File: find_best_model.py
* Author: Fan Kai
* Soochow University
* Created: 2024-03-09 11:33:13
* ----------------------------
* Modified: 2024-03-09 12:22:56
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import re

from rich.progress import BarColumn, Progress, TimeElapsedColumn

CHECKPOINTS_DIR = "/workspace/checkpoints/cvcnet-mtl-mlp_18-2"

CKP_DIR_PATTERN = r"cvcnet-mtl-mlp_\S+_BS\d+_LR\d+(\.\d+)?(e[-+]?\d+)?_EP\d+_\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}"
best_model = None
best_val_loss = float("inf")

progress = Progress(
    "[progress.description]{task.description}",
    BarColumn(),
    "[progress.percentage]{task.percentage:>3.0f}%",
    TimeElapsedColumn(),
)
task = progress.add_task("[cyan]Searching for the best model...", total=100)


for root, dirs, _ in os.walk(CHECKPOINTS_DIR):
    for dir_name in dirs:
        if re.match(CKP_DIR_PATTERN, dir_name):
            log_files = [
                file
                for file in os.listdir(os.path.join(root, dir_name))
                if file.startswith("log_")
            ]
            if log_files:
                log_file = os.path.join(root, dir_name, log_files[0])
                with open(log_file, "r") as f:
                    for line in f:
                        if "Best valloss" in line:
                            val_loss = float(line.split(":")[-1].strip())
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_model = os.path.join(root, dir_name)
            progress.update(task, advance=1)
progress.stop()

if best_model:
    print(f"Best model found: {best_model}")
    print(f"Best validation loss: {best_val_loss}")
else:
    print("No best model found.")
