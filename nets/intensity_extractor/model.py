import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.intensity_extractor.base_model import base_Model



def mixup(x_emo, x_neu):
    batch_size, seq_len_emo, channels = x_emo.size()
    seq_len_neu = x_neu.size(1)

    # 더 긴 시퀀스에 맞춰 인터폴레이션
    target_len = max(seq_len_emo, seq_len_neu)

    x_emo_interp = F.interpolate(x_emo.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(
        1, 2)
    x_neu_interp = F.interpolate(x_neu.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(
        1, 2)

    lambda_i = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
        x_emo.device)
    lambda_i = lambda_i * 0.66
    lambda_j = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
        x_emo.device)
    lambda_j = lambda_j * 0.66

    # lambda_i = torch.tensor(0.85).expand(batch_size).unsqueeze(-1).to(
    #     x_emo.device)
    # lambda_j = torch.tensor(0.15).expand(batch_size).unsqueeze(-1).to(
    #     x_emo.device)

    x_mix_i = lambda_i.unsqueeze(-1) * x_emo_interp + (1 - lambda_i.unsqueeze(-1)) * x_neu_interp
    x_mix_j = lambda_j.unsqueeze(-1) * x_emo_interp + (1 - lambda_j.unsqueeze(-1)) * x_neu_interp

    return x_mix_i, x_mix_j, lambda_i.squeeze(), lambda_j.squeeze()


class EmotionEmbedding(nn.Module):
    def __init__(self, num_emotions, embedding_dim):
        super(EmotionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_emotions, embedding_dim)

    def forward(self, emotion_class):
        return self.embedding(emotion_class)

# class IntensityExtractor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
#         super().__init__()
#         # 1. 기존 CNN feature extractor 유지
#         # self.base_model = base_Model(configs)
#
#         # 2. Transformer encoder로 시퀀스 정보 처리
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=256, nhead=8),  # base_Model의 출력이 256차원
#             num_layers=num_layers
#         )
#
#         # 3. Emotion-aware attention 추가
#         self.emotion_embed = nn.Embedding(num_emotions, 256)
#         self.emotion_attention = nn.MultiheadAttention(256, 8)
#         self.emotion_attention2 = nn.MultiheadAttention(256, 8)
#
#         # 4. Feature fusion
#         # self.fusion = nn.Sequential(
#         #     nn.Linear(512, 256),
#         #     nn.ReLU(),
#         #     nn.LayerNorm(256),
#         #     )
#         self.fusion = nn.Sequential(
#             nn.Linear(768, 512),
#             nn.GELU(),
#             nn.LayerNorm(512),
#             nn.Linear(512, 256),
#             nn.GELU(),
#             nn.LayerNorm(256),
#         )
#
#         self.fc = nn.Linear(input_dim, hidden_dim)
#
#
# #
#     def forward(self, x, emotion_class):
#         # CNN feature extraction
#         # base_features = self.base_model(x)  # [B, T, 256]
#         x = self.fc(x)
#         # Transformer processing
#         transformer_out = self.transformer(x.transpose(0, 1)).transpose(0, 1)
#
#         # Emotion-aware processing
#         emo_embed = self.emotion_embed(emotion_class).expand(-1, x.size(1), -1)
#         attn_output, _ = self.emotion_attention(
#             emo_embed.transpose(0, 1),
#             transformer_out.transpose(0, 1),
#             transformer_out.transpose(0, 1),
#         )
#
#         attn_output2, _ = self.emotion_attention2(
#             transformer_out.transpose(0, 1),
#             emo_embed.transpose(0, 1),
#             emo_embed.transpose(0, 1),
#         )
#
#         # attn_output = attn_output.transpose(0, 1) + attn_output2.transpose(0, 1)
#
#         attn = torch.cat([attn_output.transpose(0, 1), attn_output2.transpose(0, 1)], dim=-1)
#         # Feature fusion
#         concat_features = torch.cat([transformer_out, attn], dim=-1)
#         intensity_representation = self.fusion(concat_features)
#
#
#         # Get final representations
#         h_mix = intensity_representation.mean(dim=1)  # Global averaging
#
#         return h_mix, intensity_representation
class IntensityExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
        super().__init__()
        # 1. 기존 CNN feature extractor 유지
        # self.base_model = base_Model(configs)

        # 2. Transformer encoder로 시퀀스 정보 처리
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=256, nhead=8),  # base_Model의 출력이 256차원
            num_layers=num_layers
        )

        # 3. Emotion-aware attention 추가
        self.emotion_embed = nn.Embedding(num_emotions, 256)
        self.emotion_attention = nn.MultiheadAttention(256, 8)

        # 4. Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(512, 256),  # 256(CNN) + 256(emotion attention)
            nn.ReLU(),
            nn.LayerNorm(256))
        self.fc = nn.Linear(input_dim, hidden_dim)


    def forward(self, x, emotion_class):
        # CNN feature extraction
        # base_features = self.base_model(x)  # [B, T, 256]
        x = self.fc(x)
        # Transformer processing
        transformer_out = self.transformer(x.transpose(0, 1)).transpose(0, 1)

        # Emotion-aware processing
        emo_embed = self.emotion_embed(emotion_class).expand(-1, x.size(1), -1)
        attn_output, _ = self.emotion_attention(
            emo_embed.transpose(0, 1),
            transformer_out.transpose(0, 1),
            transformer_out.transpose(0, 1),
        )
        attn_output = attn_output.transpose(0, 1)

        # Feature fusion
        concat_features = torch.cat([transformer_out, attn_output], dim=-1)
        intensity_representation = self.fusion(concat_features)

        # Get final representations
        h_mix = intensity_representation.mean(dim=1)  # Global averaging
        # intensity_representation = self.intensity_proj(h_mix)

        return h_mix, intensity_representation

# class IntensityExtractor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, num_layers, num_emotions):
#         super(IntensityExtractor, self).__init__()
#         self.transformer = nn.TransformerEncoder(
#             nn.TransformerEncoderLayer(d_model=256, nhead=8),
#             num_layers=num_layers
#         )
#         self.emotion_embedding = EmotionEmbedding(num_emotions, 256)
#         self.fc = nn.Linear(input_dim, hidden_dim)
#
#     def forward(self, x, emotion_class):
#         x = self.fc(x)
#         x = x.permute(1, 0, 2)  # (seq_len, batch, features)
#         x = self.transformer(x)
#         emotion_emb = self.emotion_embedding(emotion_class).permute(1,0,2)
#         x = x + emotion_emb
#         x = x.permute(1, 2, 0)  # (batch, features, seq_len)
#         h_mix = x.mean(dim=2)
#         return h_mix, x


class RankModel(nn.Module):
    def __init__(self, configs):
        super(RankModel, self).__init__()
        self.base_model = base_Model()
        self.intensity_extractor = IntensityExtractor(82, configs['model']['hidden_dim'], configs['model']['num_layers'], configs['model']['num_emotions'])
        self.projector = nn.Sequential(nn.Linear(configs['model']['hidden_dim'], 1))

        # self.emotion_head = nn.Sequential(
        #     nn.Linear(256, 128),
        #     nn.GELU(),
        #     nn.LayerNorm(128),
        #     nn.Linear(128, 256)
        # )

    def forward(self, x_emo, x_neu, emotion_class):
        # x_i = self.base_model(x_emo)
        # x_j = self.base_model(x_neu)

        x_mix_i, x_mix_j, lambda_i, lambda_j = mixup(x_emo, x_neu)

        h_mix_i, feat_i = self.intensity_extractor(x_mix_i, emotion_class)
        h_mix_j, feat_j = self.intensity_extractor(x_mix_j, emotion_class)

        r_mix_i = self.projector(h_mix_i)
        r_mix_j = self.projector(h_mix_j)

        # emotion_logits_i = self.emotion_head(h_mix_i)
        # emotion_logits_j = self.emotion_head(h_mix_j)

        return h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j