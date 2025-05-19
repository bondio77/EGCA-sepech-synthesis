import torch
import torch.nn.functional as F
import torch.nn as nn


def mixup_loss(h_mix, y_emo, y_neu, lambda_mix):
    return lambda_mix * F.cross_entropy(h_mix, y_emo.squeeze(-1)) + (1 - lambda_mix) * F.cross_entropy(h_mix, y_neu.squeeze(-1))

def rank_loss(r_mix_i, r_mix_j, lambda_i, lambda_j, eps=1e-8):
    lambda_diff = lambda_i - lambda_j
    lambda_diff_normalized = 0.5 + (0.5 * lambda_diff)
    p_ij = torch.sigmoid(r_mix_i - r_mix_j)
    return -lambda_diff_normalized * torch.log(p_ij+eps) - (1 - lambda_diff_normalized) * torch.log(1 - p_ij+eps)

def total_loss(h_mix_i, h_mix_j, r_mix_i, r_mix_j, y_emo, y_neu, lambda_i, lambda_j, alpha, beta):
    l_mixup = mixup_loss(h_mix_i, y_emo, y_neu, lambda_i) + mixup_loss(h_mix_j, y_emo, y_neu, lambda_j)
    l_rank = rank_loss(r_mix_i.squeeze(-1), r_mix_j.squeeze(-1), lambda_i, lambda_j)
    return l_mixup, l_rank, (alpha * l_mixup + beta * l_rank).mean()