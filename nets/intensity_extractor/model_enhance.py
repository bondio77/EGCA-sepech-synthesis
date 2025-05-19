import torch
import torch.nn as nn
import torch.nn.functional as F



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



class RankModel(nn.Module):
    def __init__(self, configs):
        super(RankModel, self).__init__()
        self.intensity_extractor = IntensityExtractor(82, 256,
                                                      6, configs['model']['num_emotions'])

        self.projector = nn.Sequential(
            nn.Linear(configs['model']['hidden_dim'], configs['model']['hidden_dim'] // 2),
            nn.GELU(),
            nn.LayerNorm(configs['model']['hidden_dim'] // 2),
            nn.Linear(configs['model']['hidden_dim'] // 2, 1)
        )

        # # 새로 추가: 감정 분류 헤드
        # self.emotion_head = nn.Sequential(
        #     nn.Linear(configs['model']['hidden_dim'], configs['model']['hidden_dim'] // 2),
        #     nn.GELU(),
        #     nn.LayerNorm(configs['model']['hidden_dim'] // 2),
        #     nn.Linear(configs['model']['hidden_dim'] // 2, configs['model']['num_emotions'])
        # )

    def forward(self, x_emo, x_neu, emotion_class):
        x_mix_i, x_mix_j, lambda_i, lambda_j = mixup(x_emo, x_neu)

        h_mix_i, feat_i = self.intensity_extractor(x_mix_i, emotion_class)
        h_mix_j, feat_j = self.intensity_extractor(x_mix_j, emotion_class)

        r_mix_i = self.projector(h_mix_i)
        r_mix_j = self.projector(h_mix_j)

        # # 새로 추가: 감정 로짓 계산
        # emotion_logits_i = self.emotion_head(h_mix_i)
        # emotion_logits_j = self.emotion_head(h_mix_j)

        return h_mix_i, h_mix_j, r_mix_i, r_mix_j, lambda_i, lambda_j


class IntensityExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_emotions, dropout=0.1):
        super().__init__()
        # 입력 특징 처리
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        )

        # 트랜스포머 인코더
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim * 4,
                # dropout=dropout,
                activation='gelu'
            ),
            num_layers=num_layers
        )

        # 감정 임베딩 및 어텐션
        self.emotion_embed = nn.Embedding(num_emotions, hidden_dim)
        self.emotion_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            # dropout=dropout
        )

        # 감정 특화 컨텍스트 게이트
        self.emotion_context_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim)
            # nn.Sigmoid()
        )

        # 특징 융합
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            # nn.Dropout(dropout)
        )

    def forward(self, x, emotion_class):
        # 입력 특징 처리
        x = self.input_projection(x)  # [B, T, H]

        # 트랜스포머 처리
        transformer_out = self.transformer(x.transpose(0, 1)).transpose(0, 1)  # [B, T, H]

        # 감정 임베딩 및 어텐션
        emo_embed = self.emotion_embed(emotion_class).expand(-1, x.size(1), -1)  # [B, 1, H]
        attn_output, _ = self.emotion_attention(
            query=emo_embed.transpose(0, 1),
            key=transformer_out.transpose(0, 1),
            value=transformer_out.transpose(0, 1)
        )
        attn_output = attn_output.transpose(0, 1)  # [B, 1, H]

        # 감정 특화 컨텍스트 게이트
        # expanded_attn = attn_output.expand(-1, transformer_out.size(1), -1)  # [B, T, H]
        concat_features = torch.cat([transformer_out, attn_output], dim=-1)  # [B, T, 2H]
        gate = self.emotion_context_gate(concat_features)  # [B, T, H]

        # 게이트된 특징 융합
        gated_transformer = transformer_out * gate  # [B, T, H]
        fusion_input = torch.cat([gated_transformer, attn_output], dim=-1)  # [B, T, 2H]
        fused_features = self.feature_fusion(fusion_input)  # [B, T, H]

        # 전역 표현 및 시퀀스 표현
        global_repr = fused_features.mean(dim=1)  # [B, H]

        return global_repr, fused_features
