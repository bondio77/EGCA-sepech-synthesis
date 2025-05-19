import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.intensity_extractor.base_model import base_Model
from nets.intensity_extractor.tdnn import *
# from speechbrain.speechbrain.dataio.dataio import length_to_mask

def mixup(x_emo, x_neu):
    """
    x_emo: (batch, seq_len_emo, channels) - 비중립 데이터
    x_neu: (batch, seq_len_neu, channels) - 중립 데이터
    """
    batch_size, seq_len_emo, channels = x_emo.size()
    seq_len_neu = x_neu.size(1)

    # 최대 길이 계산
    target_len = max(seq_len_emo, seq_len_neu)

    # x_emo 패딩 (target_len보다 짧으면 0으로 패딩)
    if seq_len_emo < target_len:
        padding_size = target_len - seq_len_emo
        x_emo_padded = F.pad(x_emo, (0, 0, 0, padding_size), value=0)  # (batch, target_len, channels)
    else:
        x_emo_padded = x_emo

    # x_neu 패딩 (target_len보다 짧으면 0으로 패딩)
    if seq_len_neu < target_len:
        padding_size = target_len - seq_len_neu
        x_neu_padded = F.pad(x_neu, (0, 0, 0, padding_size), value=0)  # (batch, target_len, channels)
    else:
        x_neu_padded = x_neu

    # Beta 분포에서 가중치 샘플링
    lambda_i = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(x_emo.device)
    lambda_j = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(x_emo.device)

    # 두 입력 혼합 (패딩 부분은 0으로 유지)
    x_mix_i = lambda_i.unsqueeze(-1) * x_emo_padded + (1 - lambda_i.unsqueeze(-1)) * x_neu_padded
    x_mix_j = lambda_j.unsqueeze(-1) * x_emo_padded + (1 - lambda_j.unsqueeze(-1)) * x_neu_padded

    return x_mix_i, x_mix_j, lambda_i.squeeze(), lambda_j.squeeze()

#
# def mixup(x_emo, x_neu, emo_length, neu_length):
#     batch_size, seq_len_emo, channels = x_emo.size()
#     seq_len_neu = x_neu.size(1)
#
#     # 더 긴 시퀀스에 맞춰 인터폴레이션
#     target_len = max(seq_len_emo, seq_len_neu)
#
#     x_emo_interp = F.interpolate(x_emo.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(
#         1, 2)
#     x_neu_interp = F.interpolate(x_neu.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(
#         1, 2)
#
#     lambda_i = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
#         x_emo.device)
#     lambda_j = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
#         x_emo.device)
#
#     # lambda_i = lambda_i * 0.66
#     # lambda_j = lambda_j * 0.66
#
#     # lambda_i = torch.tensor(0.85).expand(batch_size).unsqueeze(-1).to(
#     #     x_emo.device)
#     # lambda_j = torch.tensor(0.15).expand(batch_size).unsqueeze(-1).to(
#     #     x_emo.device)
#
#     x_mix_i = lambda_i.unsqueeze(-1) * x_emo_interp + (1 - lambda_i.unsqueeze(-1)) * x_neu_interp
#     x_mix_j = lambda_j.unsqueeze(-1) * x_emo_interp + (1 - lambda_j.unsqueeze(-1)) * x_neu_interp
#
#     return x_mix_i, x_mix_j, lambda_i.squeeze(), lambda_j.squeeze()


class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions, embedding_dim):
        super(EmotionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)

    def forward(self, emotion_class):
        return self.embedding(emotion_class)


class IntensityExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
        super(IntensityExtractor, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),
            num_layers=num_layers
        )
        self.tdnn = TDNNBlock(input_dim, hidden_dim, kernel_size=5, dilation=1)
        self.se_res2blocks = nn.ModuleList([
            SERes2NetBlock(hidden_dim, 256, kernel_size=3, dilation=2),
            SERes2NetBlock(256, 256, kernel_size=3, dilation=3),
            SERes2NetBlock(256, 256, kernel_size=3, dilation=4)
        ])
        self.feature_aggregate = Conv1d(in_channels=hidden_dim * 3, out_channels=hidden_dim, kernel_size=1)
        self.emotion_embedding = EmotionEmbedding(num_emotions, 256)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim*2, hidden_dim)
        self.asp = AttentiveStatisticsPooling(hidden_dim)
        self.asp_bn = BatchNorm1d(input_size=hidden_dim*2)

    def forward(self, x, emotion_class):
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        emotion_emb = self.emotion_embedding(emotion_class).permute(1, 0, 2)
        x = x + emotion_emb
        x = x.permute(1,2,0)
        xl = []
        for layer in self.se_res2blocks:
            try:
                x = layer(x, lengths=None)
            except TypeError:
                x = layer(x)
            xl.append(x)
        x = torch.cat(xl, dim=1)
        frame_x = self.feature_aggregate(x)
        mean_x = frame_x.mean(dim=-1)
        return mean_x, frame_x


class RankModel(nn.Module):
    def __init__(self, configs):
        super(RankModel, self).__init__()
        self.base_model = base_Model()
        self.intensity_extractor = IntensityExtractor(82, configs['model']['hidden_dim'],
                                                      configs['model']['num_layers'], configs['model']['num_emotions'])
        self.projector = nn.Sequential(nn.Linear(configs['model']['hidden_dim'], 1))


def forward(self, x_emo, x_neu, emotion_class, lengths_emo=None, lengths_neu=None):
        # x_i = self.base_model(x_emo)
        # x_j = self.base_model(x_neu)

        x_mix_i, x_mix_j, lambda_i, lambda_j = mixup(x_emo, x_neu)

        h_mix_i, feat_i = self.intensity_extractor(x_mix_i, emotion_class)
        h_mix_j, feat_j = self.intensity_extractor(x_mix_j, emotion_class)

        r_mix_i = self.projector(h_mix_i)
        r_mix_j = self.projector(h_mix_j)

        return h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j