import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl

from pytorch3d.transforms import quaternion_to_matrix

from shape_assembly.models.encoder.pointnet import PointNet
from shape_assembly.models.encoder.dgcnn import DGCNN
from shape_assembly.models.baseline.transformer import Transformer
from shape_assembly.models.baseline.regressor import Regressor

class ShapeAssemblyNet(pl.LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = self.init_encoder()
        self.corr_module = self.init_corr_module()
        self.pose_predictor = self.init_pose_predictor()

    def init_encoder(self):
        if self.cfg.encoder == 'dgcnn':
            encoder = DGCNN(feat_dim=self.cfg.model.pc_feat_dim)
        elif self.cfg.encoder == 'pointnet':
            encoder = PointNet(feat_dim=self.cfg.model.pc_feat_dim)
        return encoder

    def init_corr_module(self):
        corr_module = Transformer(cfg=self.cfg)
        return corr_module

    def init_pose_predictor(self):
        pose_predictor = Regressor(pc_feat_dim=self.cfg.model.pc_feat_dim)
        return pose_predictor

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )
        return optimizer
    
    def forward(self, src_pc, tgt_pc):

        src_point_feat = self.encoder(src_pc)
        tgt_point_feat = self.encoder(tgt_pc)

        src_corr_feat = self.corr_module(src_point_feat, tgt_point_feat)
        tgt_corr_feat = self.corr_module(tgt_point_feat, src_point_feat)

        src_concat_feat = torch.cat((src_point_feat, src_corr_feat), dim=1)
        tgt_concat_feat = torch.cat((tgt_point_feat, tgt_corr_feat), dim=1)

        src_feat = torch.max(src_concat_feat, dim=2)[0]
        tgt_feat = torch.max(tgt_concat_feat, dim=2)[0]

        src_quat, src_trans = self.pose_predictor(src_feat)
        tgt_quat, tgt_trans = self.pose_predictor(tgt_feat)

        pred_dict = {
            'src_quat': src_quat,
            'src_trans': src_trans,
            'tgt_quat': tgt_quat,
            'tgt_trans': tgt_trans,
        }

        return pred_dict

    def compute_point_loss(self, batch_data, pred_data):
        # Point clouds
        src_pc = batch_data['src_pc'] # batch x 3 x 1024
        tgt_pc = batch_data['tgt_pc'] # batch x 3 x 1024

        # Ground truths
        src_quat_gt  = batch_data['src_quat']
        tgt_quat_gt  = batch_data['tgt_quat']
        src_trans_gt = batch_data['src_trans']
        tgt_trans_gt = batch_data['tgt_trans']

        # Model predictions
        src_quat_pred  = pred_data['src_quat']
        tgt_quat_pred  = pred_data['tgt_quat']
        src_trans_pred = pred_data['src_trans']
        tgt_trans_pred = pred_data['tgt_trans']

        # Source point loss
        transformed_src_pc_pred = quaternion_to_matrix(src_quat_pred) @ src_pc + src_trans_pred # batch x 3 x 1024
        transformed_src_pc_gt   = quaternion_to_matrix(src_quat_gt) @ src_pc + src_trans_gt     # batch x 3 x 1024
        src_point_loss          = torch.mean(torch.sum((transformed_src_pc_pred - transformed_src_pc_gt) ** 2, axis=1))

        # Target point loss
        transformed_tgt_pc_pred = quaternion_to_matrix(tgt_quat_pred) @ tgt_pc + tgt_trans_pred # batch x 3 x 1024
        transformed_tgt_pc_gt   = quaternion_to_matrix(tgt_quat_gt) @ tgt_pc + tgt_trans_gt     # batch x 3 x 1024
        tgt_point_loss          = torch.mean(torch.sum((transformed_tgt_pc_pred - transformed_tgt_pc_gt) ** 2, axis=1))
        
        # Point loss
        point_loss = (src_point_loss + tgt_point_loss) / 2.0
        return point_loss

    def compute_quat_loss(self, batch_data, pred_data):
        # Ground truths
        src_quat_gt = batch_data['src_quat']
        tgt_quat_gt = batch_data['tgt_quat']

        # Model predictions
        src_quat_pred = pred_data['src_quat']
        tgt_quat_pred = pred_data['tgt_quat']

        # Compute loss
        src_quat_loss = F.mse_loss(src_quat_pred, src_quat_gt)
        tgt_quat_loss = F.mse_loss(tgt_quat_pred, tgt_quat_gt)
        quat_loss     = (src_quat_loss + tgt_quat_loss) / 2.0
        return rot_param_loss

    def compute_trans_loss(self, batch_data, pred_data):
        # Ground truths
        src_trans_gt = batch_data['src_trans'] # batch x 3 x 1
        tgt_trans_gt = batch_data['tgt_trans'] # batch x 3 x 1

        # Model predictions
        src_trans_pred = pred_data['src_trans'] # batch x 3 x 1
        tgt_trans_pred = pred_data['tgt_trans'] # batch x 3 x 1

        # Compute loss
        src_trans_loss = F.mse_loss(src_trans_pred, src_trans_gt)
        tgt_trans_loss = F.mse_loss(tgt_trans_pred, tgt_trans_gt)
        trans_loss     = (src_trans_loss + tgt_trans_loss) / 2.0
        return trans_loss

    def training_step(self, batch_data, batch_idx):
        total_loss = self.forward_pass(batch_data, mode='train')
        return total_loss

    def validation_step(self, batch_data, batch_idx):
        total_loss = self.forward_pass(batch_data, mode='val')
        return total_loss

    def forward_pass(self, batch_data, mode):
        # Forward pass
        pred_data  = self.forward(batch_data)

        # Point loss
        point_loss = self.compute_point_loss(batch_data, pred_data)

        # Pose loss
        quat_loss  = self.compute_quat_loss(batch_data, pred_data)
        trans_loss = self.compute_trans_loss(batch_data, pred_data)

        # Total loss
        total_loss = point_loss + quat_loss + trans_loss

        # Logger
        self.log("{}/total loss".format(mode), total_loss, logger=True)
        self.log("{}/point loss".format(mode), point_loss, logger=True)
        self.log("{}/quat loss".format(mode),  quat_loss,  logger=True)
        self.log("{}/trans loss".format(mode), trans_loss, logger=True)

        return total_loss
