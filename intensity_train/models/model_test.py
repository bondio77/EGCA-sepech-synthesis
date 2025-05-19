import torch
import torch.nn as nn
import torch.nn.functional as F
from emo_intensity.models.base_model import base_Model



def mixup(x_emo, x_neu):
    batch_size, seq_len_emo, channels = x_emo.size()
    seq_len_neu = x_neu.size(1)

    # 더 긴 시퀀스에 맞춰 인터폴레이션
    target_len = max(seq_len_emo, seq_len_neu)

    x_emo_interp = F.interpolate(x_emo.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(
        1, 2)
    x_neu_interp = F.interpolate(x_neu.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(
        1, 2)

    # lambda_i = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
    #     x_emo.device)
    lambda_jj = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
        x_emo.device)

    lambda_i = torch.tensor(0.85).expand(batch_size).unsqueeze(-1).to(
         x_emo.device)
    lambda_j = torch.tensor(0.15).expand(batch_size).unsqueeze(-1).to(
         x_emo.device)

    x_mix_i = lambda_i.unsqueeze(-1) * x_emo_interp + (1 - lambda_i.unsqueeze(-1)) * x_neu_interp
    x_mix_j = lambda_j.unsqueeze(-1) * x_emo_interp + (1 - lambda_j.unsqueeze(-1)) * x_neu_interp

    return x_mix_i, x_mix_j, lambda_i.squeeze(), lambda_j.squeeze()


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
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=8),
            num_layers=num_layers
        )
        self.emotion_embedding = EmotionEmbedding(num_emotions, input_dim) # nn.Embedding
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, emotion_class):
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        emotion_emb = self.emotion_embedding(emotion_class).permute(1,0,2)
        x = x + emotion_emb
        x = x.permute(1, 2, 0)  # (batch, features, seq_len)
        h_mix = x.mean(dim=2) # average seq_len dimension = (batch, features)
        return h_mix, x


class RankModel(nn.Module):
    def __init__(self, configs):
        super(RankModel, self).__init__()
        self.base_model = base_Model(configs)
        self.intensity_extractor = IntensityExtractor(256, configs['hidden_dim'], configs['num_layers'], configs['num_emotions'])
        self.projector = nn.Linear(configs['hidden_dim'], 1)

    def forward(self, x_emo, x_neu, emotion_class):
        x_i = self.base_model(x_emo)
        x_j = self.base_model(x_neu)

        # x_mix_i, x_mix_j, lambda_i, lambda_j = mixup(x_i, x_j)

        h_mix_i, feat_i = self.intensity_extractor(x_i, emotion_class)
        h_mix_j, feat_j = self.intensity_extractor(x_j, emotion_class)

        r_mix_i = self.projector(h_mix_i)
        r_mix_j = self.projector(h_mix_j)

        return r_mix_i, r_mix_j