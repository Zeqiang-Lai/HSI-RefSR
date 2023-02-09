import torch.nn as nn
import torch
import torch.nn.functional as F

from . import blocks


class MaskPredictor(nn.Module):
    def __init__(self, input_dim, project_dim, offset_feat_dim,
                 num_offset_feat_extractor_res=1, num_weight_predictor_res=1, use_offset=True, offset_modulo=1.0,
                 use_bn=False, activation='relu'):
        super().__init__()
        self.use_offset = use_offset
        self.offset_modulo = offset_modulo
        
        self.feat_project_layer = blocks.conv_block(input_dim, project_dim, 1, stride=1, padding=0,
                                                    batch_norm=use_bn,
                                                    activation=activation)

        offset_feat_extractor = []
        offset_feat_extractor.append(blocks.conv_block(2, offset_feat_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                                       activation=activation))

        for _ in range(num_offset_feat_extractor_res):
            offset_feat_extractor.append(blocks.ResBlock(offset_feat_dim, offset_feat_dim, stride=1,
                                                         batch_norm=use_bn, activation=activation))
        self.offset_feat_extractor = nn.Sequential(*offset_feat_extractor)
        
        
        weight_predictor = []
        weight_predictor.append(blocks.conv_block(project_dim * 2 + offset_feat_dim * use_offset, 2 * project_dim, 3,
                                                  stride=1, padding=1, batch_norm=use_bn, activation=activation))

        for _ in range(num_weight_predictor_res):
            weight_predictor.append(blocks.ResBlock(2 * project_dim, 2 * project_dim, stride=1,
                                                    batch_norm=use_bn, activation=activation))

        weight_predictor.append(blocks.conv_block(2 * project_dim, 1, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)
        

    def forward(self, x, ref, flow):
        x_proj = self.feat_project_layer(x)
        ref_proj = self.feat_project_layer(ref)
        # flow_feat = self.offset_feat_extractor(flow)
        flow_feat = self.offset_feat_extractor(flow % self.offset_modulo)
        weight_pred_in = [x_proj, ref_proj, flow_feat]
        weight_pred_in = torch.cat(weight_pred_in, dim=1)
        weight = self.weight_predictor(weight_pred_in)
        weight_norm = torch.sigmoid(weight)
        return weight_norm