import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from math import floor, log, pi

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
    lambda_j = torch.distributions.Beta(torch.tensor([1.0]), torch.tensor([1.0])).sample((batch_size,)).to(
        x_emo.device)

    lambda_i = lambda_i * 0.66
    lambda_j = lambda_j * 0.66

    # lambda_i = torch.tensor(0.85).expand(batch_size).unsqueeze(-1).to(
    #     x_emo.device)
    # lambda_j = torch.tensor(0.15).expand(batch_size).unsqueeze(-1).to(
    #     x_emo.device)

    x_mix_i = lambda_i.unsqueeze(-1) * x_emo_interp + (1 - lambda_i.unsqueeze(-1)) * x_neu_interp
    x_mix_j = lambda_j.unsqueeze(-1) * x_emo_interp + (1 - lambda_j.unsqueeze(-1)) * x_neu_interp

    return x_mix_i, x_mix_j, lambda_i.squeeze(), lambda_j.squeeze()



"""
Continuous Time Positional Embedding
"""
class LearnedPositionalEmbedding(nn.Module):
    """Used for continuous time"""
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        # 기존 코드에서 `x`가 [batch_size, seq_len]일 경우를 고려하여 변환
        x = rearrange(x, "b t -> b t 1")  # [B, T, 1]

        # 주파수 임베딩 적용
        freqs = x * rearrange(self.weights, "d -> 1 1 d") * 2 * pi  # [B, T, D]

        # 사인, 코사인 변환 및 concat
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)  # [B, T, 2D]

        # 원래 입력과 결합
        fouriered = torch.cat((x, fouriered), dim=-1)  # [B, T, 2D+1]

        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
        nn.GELU(),
    )

"""
Adaptive Layer Normalization (Emotion-Aware Normalization)
"""
class AdaLayerNorm(nn.Module):
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, s):

        h = self.fc(s)
        h = h.view(h.size(0), -1, 1)
        gamma, beta = torch.chunk(h, chunks=2, dim=1)
        gamma, beta = gamma.transpose(1, -1), beta.transpose(1, -1)

        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        x = (1 + gamma) * x + beta  # Adaptive scaling

        return x

"""
Style Transformer Block with Continuous Time Positional Embedding
"""
class StyleTransformerBlock(nn.Module):
    def __init__(self, features, num_heads, head_features, style_dim, multiplier, time_dim):
        super().__init__()

        self.attention = nn.MultiheadAttention(embed_dim=features, num_heads=num_heads)
        self.norm1 = AdaLayerNorm(style_dim, features)
        self.norm2 = nn.LayerNorm(features)

        self.feed_forward = nn.Sequential(
            nn.Linear(features, features * multiplier),
            nn.GELU(),
            nn.Linear(features * multiplier, features),
        )

        # **Continuous Time Positional Embedding 적용**
        self.to_time = TimePositionalEmbedding(time_dim, features)

    def forward(self, x, s, time):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # **Continuous Time Positional Embedding 추가**
        time_emb = self.to_time(time).to(device)  # [batch, seq_len, features]
        x = x + time_emb  # 위치 정보 추가

        # Adaptive Normalization
        x = self.norm1(x, s)
        x_attn, _ = self.attention(x, x, x)
        x = x + x_attn
        x = self.norm2(x)

        # Feed Forward Network
        x_ffn = self.feed_forward(x)
        x = x + x_ffn

        return x


"""
Style Transformer with Emotion & Continuous Time Embedding
"""
class StyleTransformer1d(nn.Module):
    def __init__(self, num_layers, channels, num_heads, head_features, multiplier, context_features, time_dim):
        super().__init__()

        self.blocks = nn.ModuleList([
            StyleTransformerBlock(
                features=channels,
                num_heads=num_heads,
                head_features=head_features,
                style_dim=context_features,
                multiplier=multiplier,
                time_dim=time_dim,  # Continuous Time Positional Embedding 추가
            ) for _ in range(num_layers)
        ])

        self.to_out = nn.Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x, time, embedding, features):
        mapping = embedding.expand(-1, x.size(1), -1)

        for block in self.blocks:
            x = block(x + mapping, features, time)  # Inject style features + time positional embedding

        # x = x.mean(dim=1).unsqueeze(1)
        x = self.to_out(x.transpose(-1, -2)).transpose(-1, -2)

        return x


"""
IntensityExtractor with CNN + Adaptive Normalization + Transformer + Continuous Time Embedding
"""
class IntensityExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_emotions, num_layers=3, time_dim=32):
        super().__init__()

        # CNN-based Feature Extraction (Multi-Scale)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(input_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(input_dim, hidden_dim, kernel_size=7, padding=3)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Adaptive Layer Normalization
        self.ada_layer_norm = AdaLayerNorm(style_dim=hidden_dim, channels=hidden_dim)

        # Style Transformer with Continuous Time Embedding
        self.style_transformer = StyleTransformer1d(
            num_layers=num_layers,
            channels=hidden_dim,
            num_heads=4,
            head_features=32,
            multiplier=2,
            context_features=hidden_dim,
            time_dim=time_dim,  # Continuous Time Positional Embedding 적용
        )

        # Emotion Embedding
        self.emotion_embed = nn.Embedding(5, hidden_dim)

        # Fully Connected Projection Layer
        self.fc = nn.Linear(hidden_dim, hidden_dim)

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, emotion_class):
        batch_size, seq_len, _ = x.shape
        device = x.device

        # **🚀 time 변수 생성 (StyleTransformer1d와 호환)**
        time = torch.linspace(0, 1, steps=seq_len, device=device)  # [seq_len]
        time = time.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]

        x = x.transpose(1, 2)
        conv_out = F.relu(self.bn1(self.conv1(x))) + F.relu(self.bn2(self.conv2(x))) + F.relu(self.bn3(self.conv3(x)))
        emotion_emb = self.emotion_embed(emotion_class)
        conv_out = self.ada_layer_norm(conv_out.transpose(1,2), emotion_emb)

        transformer_out = self.style_transformer(conv_out, time, embedding=emotion_emb, features=emotion_emb)
        frame_output = transformer_out
        global_output = frame_output.mean(dim=1)

        return global_output, frame_output



class RankModel(nn.Module):
    def __init__(self, configs):
        super(RankModel, self).__init__()
        # self.base_model = base_Model(configs)
        self.intensity_extractor = IntensityExtractor(82, configs['model']['hidden_dim'], configs['model']['num_layers'], configs['model']['num_emotions'])
        # self.projector = nn.Sequential(nn.Linear(configs['hidden_dim'], 1), nn.Sigmoid())
        self.projector = nn.Sequential(nn.Linear(configs['model']['hidden_dim'], 1))

    def forward(self, x_emo, x_neu, emotion_class):
        # x_i = self.base_model(x_emo)
        # x_j = self.base_model(x_neu)

        x_mix_i, x_mix_j, lambda_i, lambda_j = mixup(x_emo, x_neu)

        h_mix_i, feat_i = self.intensity_extractor(x_mix_i, emotion_class)
        h_mix_j, feat_j = self.intensity_extractor(x_mix_j, emotion_class)

        r_mix_i = self.projector(h_mix_i)
        r_mix_j = self.projector(h_mix_j)

        return h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j