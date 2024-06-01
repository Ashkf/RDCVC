"""
* Script for running the GA algorithm
* Once organized, extract the logic to
*   "src/RDCVC/models/optimization/utils/GARunner.py"
*
* File: optimize_GA.py
* Author: Fan Kai
* Soochow University
* Created: 2024-01-02 03:12:29
* ----------------------------
* Modified: 2024-03-20 15:21:42
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
# 导入自定义运行接口
from RDCVC.models.optimization.GA.SN1.run import run as run_SN1  # noqa: E402
from RDCVC.models.optimization.utils.Args import (  # noqa: E402
    ArgsManagerMode,
    GAArgsManager,
)


def results_postprocess(self):
    """
    docstring
    """
    pass


if __name__ == "__main__":
    # ------------------------- Args ------------------------- #
    argMan = GAArgsManager(ArgsManagerMode.CLI_FIRST)
    argMan.get_args()
    argMan.save_args()
    args = argMan.args
    # -------------------------- run ------------------------- #

    run_SN1(args=args)
