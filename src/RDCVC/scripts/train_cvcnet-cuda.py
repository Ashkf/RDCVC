"""
* train script for CVCNet
*
* File: train.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 11:15:29
* ----------------------------
* Modified: 2023-12-03 12:50:26
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

# import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from RDCVC.models.prediciton.CVCNet.utils.trainer import Trainer  # noqa: E402

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

    # with torch.autograd.profiler.profile(
    #     enabled=True, use_cuda=True, record_shapes=False, profile_memory=False
    # ) as prof:
    #     trainer.train()
    # print(prof.table())
    # prof.export_chrome_trace("/cvcnet_profile_20231203.json")
