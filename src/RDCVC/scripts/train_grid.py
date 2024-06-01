"""
*
*
* File: train_grid.py
* Author: Fan Kai
* Soochow University
* Created: 2024-02-10 17:04:51
* ----------------------------
* Modified: 2024-02-26 15:02:35
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import itertools
import subprocess

from tqdm import tqdm


def run_script(args):
    cmd = ["python", "src/RDCVC/scripts/train_cvcnet-cuda.py"] + args
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    # 设定参数网格
    param_combs = list(
        itertools.product(
            num_e_layers := [1, 2, 3, 4, 5],
            num_tasks_experts := [1, 2, 3, 4, 5],
            num_shared_experts := [1, 2, 3, 4, 5],
            batch_size := [16, 32, 64],
            LWS := ["none", "DWA"],
        )
    )

    progress_bar = tqdm(total=len(param_combs), desc="Training")
    for comb in param_combs:
        num_e_layers, num_tasks_experts, num_shared_experts, batch_size, LWS = comb

        # Construct argument list for subprocess
        cmd_args = [
            f"cvcnet-mtl-mlp_18_{str(num_e_layers)}_{str(num_tasks_experts)}_{str(num_shared_experts)}_64-64-64_64-64-64",
            "cvcnet",
            "--train_path=data/train/rdc_data_cleaned_train.csv",
            "--eval_path=data/test/rdc_data_cleaned_eval.csv",
            "--seed=42",
            "--lr=1e-2",
            "--lr_scheduler",
            "--lrsmin=1e-5",
            f"--batch_size={batch_size}",
            "--epochs=10000",
            "--save_every=100",
            "--normalize_target=xy",
            "--normalize_method=zscore",
            "--earlystop",
            "--espatience=100",
            f"--LWS={LWS}",
            "--DWA_limit=0.1-10",
            "--DWA_weight=1,1",
        ]

        run_script(cmd_args)

        progress_bar.update(1)
        progress_bar.set_postfix({"Combination": comb})
    progress_bar.close()
