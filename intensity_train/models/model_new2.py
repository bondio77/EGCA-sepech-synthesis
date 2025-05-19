import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum, pi

from torch import Tensor


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

class AdaLayerNorm(nn.Module):
    """ Adaptive Layer Normalization (Emotion-Aware Normalization) """
    def __init__(self, style_dim, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.fc = nn.Linear(style_dim, channels * 2)

    def forward(self, x, style):
        h = self.fc(style)  # [B, 2*C]
        gamma, beta = torch.chunk(h, chunks=2, dim=-1)  # [B, C] each
        gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)  # [B, 1, C]
        x = F.layer_norm(x, (self.channels,), eps=self.eps)
        return (1 + gamma) * x + beta


class AttentionBase(nn.Module):
    """ Standard Self-Attention """
    def __init__(self, features, head_features, num_heads):
        super().__init__()
        self.scale = head_features ** -0.5
        self.num_heads = num_heads
        mid_features = head_features * num_heads
        self.to_out = nn.Linear(mid_features, features)

    def forward(self, q, k, v):
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (q, k, v))
        attn = einsum("b h n d, b h t d -> b h n t", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = einsum("b h n t, b h t d -> b h n d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class StyleAttention(nn.Module):
    """ Emotion-Driven Attention """
    def __init__(self, features, style_dim, head_features, num_heads, context_features):
        super().__init__()
        self.norm = AdaLayerNorm(style_dim, features)
        self.norm_context = AdaLayerNorm(style_dim, context_features)
        self.to_q = nn.Linear(features, head_features * num_heads, bias=False)
        self.to_kv = nn.Linear(context_features, head_features * num_heads * 2, bias=False)
        self.attention = AttentionBase(features, head_features, num_heads)

    def forward(self, x, style, context):
        x, context = self.norm(x, style), self.norm_context(context, style)
        q, k, v = self.to_q(x), *torch.chunk(self.to_kv(context), chunks=2, dim=-1)
        return self.attention(q, k, v)


class StyleTransformerBlock(nn.Module):
    """ Transformer Block with Self and Cross Attention """
    def __init__(self, features, num_heads, head_features, style_dim, multiplier, context_features):
        super().__init__()
        self.attention = StyleAttention(features, style_dim, head_features, num_heads, context_features)
        self.cross_attention = StyleAttention(features, style_dim, head_features, num_heads, context_features)
        self.feed_forward = nn.Sequential(
            nn.Linear(features, features * multiplier),
            nn.GELU(),
            nn.Linear(features * multiplier, features),
        )

    def forward(self, x, style, context):
        x = self.attention(x, style, context) + x
        x = self.cross_attention(x, style, context) + x
        x = self.feed_forward(x) + x
        return x


class LearnedPositionalEmbedding(nn.Module):
    """ Used for continuous time embeddings """
    def __init__(self, dim: int):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b -> b 1")
        freqs = x * rearrange(self.weights, "d -> 1 d") * 2 * pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


def TimePositionalEmbedding(dim: int, out_features: int) -> nn.Module:
    """ Time-based embedding transformation """
    return nn.Sequential(
        LearnedPositionalEmbedding(dim),
        nn.Linear(in_features=dim + 1, out_features=out_features),
        nn.GELU(),
    )


class StyleTransformer1d(nn.Module):
    """ Transformer-based Intensity Extractor """
    def __init__(self, num_layers, channels, num_heads, head_features, multiplier, context_features, style_dim):
        super().__init__()
        self.blocks = nn.ModuleList([
            StyleTransformerBlock(
                features=channels,
                num_heads=num_heads,
                head_features=head_features,
                style_dim=style_dim,
                multiplier=multiplier,
                context_features=context_features,
            )
            for _ in range(num_layers)
        ])
        self.to_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x, style, context):
        for block in self.blocks:
            x = block(x, style, context)
        x = self.to_out(x.transpose(1, 2)).transpose(1, 2)  # Conv 적용 후 차원 복원
        return x


class IntensityExtractor(nn.Module):
    """ Intensity Extractor with Transformer + Time Positional Embedding """
    def __init__(self, input_dim, style_dim, hidden_dim=256, num_heads=8, num_layers=3, time_embedding_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.style_dim = style_dim

        # 🔹 Conv1D Layers (Feature Extraction)
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)

        # 🔹 Positional Embedding
        self.time_embedding = TimePositionalEmbedding(time_embedding_dim, hidden_dim)

        # 🔹 Transformer-based Processing
        self.style_transformer = StyleTransformer1d(
            num_layers=num_layers,
            channels=hidden_dim,
            num_heads=num_heads,
            head_features=hidden_dim // num_heads,
            multiplier=4,
            context_features=hidden_dim,
            style_dim=style_dim
        )

        # 🔹 Final Intensity Prediction Layers
        self.fc_seq = nn.Linear(hidden_dim, 1)  # Time Sequence Output
        self.fc_mean = nn.Linear(hidden_dim, 1)  # Global Average Pooling Output

    def forward(self, x, emotion_emb):
        """
        x: [batch, time, input_dim]  (Audio Features)
        emotion_emb: [batch, style_dim] (Emotion Style Embedding)
        time_steps: [batch] (Time Step Embeddings)
        """
        B, T, _ = x.shape  # [B, T, C]
        batch_size, seq_len, _ = x.shape
        device = x.device
        # 1 Convolutional Feature Extraction
        x = x.transpose(1, 2)  # [B, C, T]
        xs = F.relu(self.conv1(x))
        xs = F.relu(self.conv2(xs))
        xs = F.relu(self.conv3(xs))
        xs = xs.transpose(1, 2)  # [B, T, C]
        time = torch.linspace(0, 1, steps=seq_len, device=device)  # [seq_len]
        time = time.unsqueeze(0).expand(batch_size, -1)  # [batch_size, seq_len]
        # 2 Time Positional Encoding
        time_emb = self.time_embedding(time)  # [B, C]
        xs = xs + time_emb.unsqueeze(1)

        # 3 Transformer Processing
        xs = self.style_transformer(x, emotion_emb, xs)

        # 4 Intensity Prediction (Sequence & Mean)
        intensity_seq = self.fc_seq(xs)  # [B, T, 1]
        intensity_mean = self.fc_mean(xs.mean(dim=1))  # [B, 1]

        return intensity_seq, intensity_mean  # Return both outputs
class RankModel(nn.Module):
    def __init__(self, configs):
        super(RankModel, self).__init__()
        # self.base_model = base_Model(configs)
        self.intensity_extractor = IntensityExtractor(82, configs['hidden_dim'], configs['num_layers'], configs['num_emotions'])
        # self.projector = nn.Sequential(nn.Linear(configs['hidden_dim'], 1), nn.Sigmoid())
        self.projector = nn.Sequential(nn.Linear(configs['hidden_dim'], 1))

    def forward(self, x_emo, x_neu, emotion_class):
        # x_i = self.base_model(x_emo)
        # x_j = self.base_model(x_neu)

        x_mix_i, x_mix_j, lambda_i, lambda_j = mixup(x_emo, x_neu)

        h_mix_i, feat_i = self.intensity_extractor(x_mix_i, emotion_class)
        h_mix_j, feat_j = self.intensity_extractor(x_mix_j, emotion_class)

        r_mix_i = self.projector(h_mix_i)
        r_mix_j = self.projector(h_mix_j)

        return h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j