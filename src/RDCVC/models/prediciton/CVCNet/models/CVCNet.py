"""
*
*
* File: cpn_mtl_mlp.py
* Author: Fan Kai
* Soochow University
* Created: 2023-11-19 03:04:08
* ----------------------------
* Modified: 2023-11-20 02:16:33
* Modified By: Fan Kai
* ========================================================================
* HISTORY:
"""

from torch import nn

from .submodules import MLPBlock


class CVCNet(nn.Module):
    """
    CPN: Cleanroom pressure Prediction Control neural Network
    多任务学习、多层感知机
    """

    def __init__(
        self, backbone_depth: int, backbone_dim: int, head_depth: int, head_dim: int
    ):
        super().__init__()
        self.backbone = MLPBlock(
            in_dim=18,
            out_dim=backbone_dim,
            hidden_dim=backbone_dim,
            num_layers=backbone_depth,
        )
        # ------------------- head tasks -------------------
        self.totvol_fresh = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.totvol_supply = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.totvol_exhaust = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.totvol_return = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.rm_pres_1 = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.rm_pres_2 = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.rm_pres_3 = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.rm_pres_4 = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.rm_pres_5 = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )
        self.rm_pres_6 = MLPBlock(
            in_dim=backbone_dim, out_dim=1, hidden_dim=head_dim, num_layers=head_depth
        )

    def forward(self, x):
        shared_features = self.backbone(x)
        out1 = self.totvol_fresh(shared_features)
        out2 = self.totvol_supply(shared_features)
        out3 = self.totvol_exhaust(shared_features)
        out4 = self.totvol_return(shared_features)
        out5 = self.rm_pres_1(shared_features)
        out6 = self.rm_pres_2(shared_features)
        out7 = self.rm_pres_3(shared_features)
        out8 = self.rm_pres_4(shared_features)
        out9 = self.rm_pres_5(shared_features)
        out10 = self.rm_pres_6(shared_features)
        return out1, out2, out3, out4, out5, out6, out7, out8, out9, out10
