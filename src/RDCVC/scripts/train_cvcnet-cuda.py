"""
* train script for CVCNet
*
* File: train.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 11:15:29
* ----------------------------
* Modified: 2024-02-06 22:39:36
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from RDCVC.models.prediciton.CVCNet.utils.trainer import Trainer

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
