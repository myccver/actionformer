from .blocks import MaskedConv1D, Scale, LayerNorm
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import (get_sinusoid_encoding, TransformerBlock, MaskedConv1D,
                     ConvBlock, LayerNorm)


class Reliabilty_Aware_Block(nn.Module):
    def __init__(self, input_dim, dropout, num_heads=8, dim_feedforward=128, pos_embed=False):
        super(Reliabilty_Aware_Block, self).__init__()
        # self.conv_query = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        # self.conv_key = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        # self.conv_value = nn.Conv1d(input_dim, input_dim, kernel_size=1, stride=1, padding=0)
        self.conv_query = MaskedConv1D(in_channels=input_dim,out_channels=input_dim,kernel_size=1,stride=1,padding=0)
        self.conv_key = MaskedConv1D(in_channels=input_dim,out_channels=input_dim,kernel_size=1,stride=1,padding=0)
        self.conv_value = MaskedConv1D(in_channels=input_dim,out_channels=input_dim,kernel_size=1,stride=1,padding=0)

        self.self_atten = nn.MultiheadAttention(input_dim, num_heads=num_heads, dropout=0.1) # 0.1
        self.linear1 = nn.Linear(input_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, input_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, features, key_padding_mask=None,):
        src = features.permute(2, 0, 1)
        q = k = src
        q = self.conv_query(features, key_padding_mask)[0].permute(2, 0, 1)
        k = self.conv_key(features, key_padding_mask)[0].permute(2, 0, 1)

        src2, attn = self.self_atten(q, k, src, key_padding_mask=~(key_padding_mask.squeeze(1)))
        src2 = src2 * key_padding_mask.permute(2, 0, 1).detach()

        src = src + self.dropout1(src2)
        # src = self.norm1(src)
        src = src * key_padding_mask.permute(2, 0, 1).detach()
        # # src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # src2 = self.linear2(self.dropout(F.relu(self.linear1(src)))* key_padding_mask.permute(2, 0, 1).detach()) * key_padding_mask.permute(2, 0, 1).detach()
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        #
        # src = src * key_padding_mask.permute(2, 0, 1).detach()

        src = src.permute(1, 2, 0)
        return src, attn







class Encoder(nn.Module):
    def __init__(self, dataset, feature_dim, RAB_args):
        super(Encoder, self).__init__()
        self.dataset = dataset
        self.feature_dim = feature_dim

        RAB_args = RAB_args
        self.RAB = nn.ModuleList([
            Reliabilty_Aware_Block(
                input_dim=self.feature_dim,
                dropout=RAB_args['drop_out'],
                num_heads=RAB_args['num_heads'],
                dim_feedforward=RAB_args['dim_feedforward'])
            for i in range(RAB_args['layer_num'])
        ])

        # self.RAB = nn.ModuleList()
        # for idx in range(RAB_args['layer_num']):
        #     self.RAB.append(
        #         TransformerBlock(
        #             self.feature_dim, RAB_args['num_heads'],
        #             n_ds_strides=(1, 1),
        #             attn_pdrop=0.0,
        #             proj_pdrop=0.0,
        #             path_pdrop=0.1,
        #             mha_win_size=19,
        #             use_rel_pe=False
        #         )
        #     )

        self.feature_embedding = nn.Sequential(
            # nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )


    def forward(self, input_features, input_mask=None, prototypes=None):

        B, T, F = input_features.transpose(1,2).shape
        # input_features                        #[B,F,T]
        prototypes = prototypes.unsqueeze(1).to(input_features.device)          #[C,1,F]
        prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]
        # 创建全1的mask，shape为 [2, 1, 60]
        ones_padding = torch.ones(prototypes.shape[0], 1, prototypes.shape[-1], dtype=input_mask.dtype, device=input_mask.device)

        # 拼接在尾部
        input_mask_extended = torch.cat([input_mask, ones_padding], dim=2)
        if hasattr(self, 'RAB'):
            layer_features = torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
            # layer_features = input_features  # [B,F,T]
            for layer in self.RAB:
                layer_features, _ = layer(layer_features, input_mask_extended)
            input_features = layer_features[:, :, :T]                           #[B,F,T]
            # input_features = layer_features  # [B,F,T]

        # if hasattr(self, 'RAB'):
        #
        #     layer_features = input_features  # [B,F,T]
        #     for layer in self.RAB:
        #         layer_features, _ = layer(layer_features, input_mask)
        #     input_features = layer_features[:, :, :T]                           #[B,F,T]
        embeded_features = input_features * input_mask.detach()
        import torch.nn.functional as F  # ✅ 恢复 F
        embeded_features = F.relu(embeded_features)

        return embeded_features



# class Encoder(nn.Module):
#     def __init__(self, dataset, feature_dim, RAB_args):
#         super(Encoder, self).__init__()
#         self.dataset = dataset
#         self.feature_dim = feature_dim
#
#         RAB_args = RAB_args
#         self.RAB = nn.ModuleList([
#             Reliabilty_Aware_Block(
#                 input_dim=self.feature_dim,
#                 dropout=RAB_args['drop_out'],
#                 num_heads=RAB_args['num_heads'],
#                 dim_feedforward=RAB_args['dim_feedforward'])
#             for i in range(RAB_args['layer_num'])
#         ])
#
#         # self.RAB = nn.ModuleList()
#         # for idx in range(RAB_args['layer_num']):
#         #     self.RAB.append(
#         #         TransformerBlock(
#         #             self.feature_dim, RAB_args['num_heads'],
#         #             n_ds_strides=(1, 1),
#         #             attn_pdrop=0.0,
#         #             proj_pdrop=0.0,
#         #             path_pdrop=0.1,
#         #             mha_win_size=19,
#         #             use_rel_pe=False
#         #         )
#         #     )
#
#         self.feature_embedding = nn.Sequential(
#             # nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#             # nn.ReLU(),
#             nn.Conv1d(in_channels=self.feature_dim, out_channels=self.feature_dim, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#         )
#
#
#     def forward(self, input_features, input_mask=None, prototypes=None, prototype_mask=None):
#
#         B, T, F = input_features.transpose(1,2).shape
#         # input_features                        #[B,F,T]
#         prototypes = prototypes.unsqueeze(1).to(input_features.device)          #[C,1,F]
#         prototypes = prototypes.view(1,F,-1).expand(B,-1,-1)                    #[B,F,C]
#         # 创建全1的mask，shape为 [2, 1, 60]
#         ones_padding = prototype_mask.unsqueeze(1).to(device=input_mask.device,dtype=input_mask.dtype)
#         # ones_padding = torch.ones(prototypes.shape[0], 1, prototypes.shape[-1], dtype=input_mask.dtype, device=input_mask.device)
#
#         # 拼接在尾部
#         input_mask_extended = torch.cat([input_mask, ones_padding], dim=2)
#         if hasattr(self, 'RAB'):
#             layer_features = torch.cat([input_features, prototypes], dim=2)     #[B,F,T+C]
#             # layer_features = input_features  # [B,F,T]
#             for layer in self.RAB:
#                 layer_features, _ = layer(layer_features, input_mask_extended)
#             input_features = layer_features[:, :, :T]                           #[B,F,T]
#             # input_features = layer_features  # [B,F,T]
#
#         # if hasattr(self, 'RAB'):
#         #
#         #     layer_features = input_features  # [B,F,T]
#         #     for layer in self.RAB:
#         #         layer_features, _ = layer(layer_features, input_mask)
#         #     input_features = layer_features[:, :, :T]                           #[B,F,T]
#         embeded_features = input_features * input_mask.detach()
#         import torch.nn.functional as F  # ✅ 恢复 F
#         embeded_features = F.relu(embeded_features)
#
#         return embeded_features