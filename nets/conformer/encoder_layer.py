#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""

import torch

from torch import nn

from nets.transformer.layer_norm import LayerNorm
from tts.meta_style.blocks import StyleAdaptiveLayerNorm
from mamba_ssm import Mamba
from nets.mamba.bimamba import Mamba as BiMamba


class EncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,   # attention_dim
        self_attn,  # encoder_selfattn_layer(*encoder_selfattn_layer_args)
        feed_forward,   # positionwise_layer(*positionwise_layer_args),
        feed_forward_macaron,   # positionwise_layer(*positionwise_layer_args) if macaron_style else None,
        conv_module,    # convolution_layer(*convolution_layer_args) if use_cnn_module else None,
        dropout_rate,   # dropout_rate,
        normalize_before=True,  # normalize_before,
        concat_after=False,     # concat_after,
        stochastic_depth_rate=0.0,  # stochastic_depth_rate * float(1 + lnum) / num_blocks,
        use_conditional_normalize=False,
    ):
        """Construct an EncoderLayer object."""
        super(EncoderLayer, self).__init__()
        # self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.use_conditional_normalize = use_conditional_normalize
        self.mamba = BiMamba(
            d_model=384,
            bimamba_type='v2')
        if use_conditional_normalize:
            self.norm_ff = StyleAdaptiveLayerNorm(size, size)
            self.norm_mha = StyleAdaptiveLayerNorm(size, size)
        else:
            self.norm_ff = LayerNorm(size)  # for the FNN module
            self.norm_mha = LayerNorm(size)  # for the MHA module

        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0

        if self.conv_module is not None:
            self.norm_conv = LayerNorm(size)  # for the CNN module

            if use_conditional_normalize:
                self.norm_final = StyleAdaptiveLayerNorm(size, size)
            else:
                self.norm_final = LayerNorm(size)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, style=None, cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask, style
            return x, mask, style

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            if self.use_conditional_normalize:
                x = self.norm_mha(x, style)
            else:
                x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            # x_att = self.self_attn(x_q, x, x, pos_emb, mask)
            x_att = self.mamba(x)

        else:
            # x_att = self.self_attn(x_q, x, x, mask)
            x_att = self.mamba(x)
        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            if self.use_conditional_normalize:
                x = self.norm_mha(x, style)
            else:
                x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            if self.use_conditional_normalize:
                x = self.norm_ff(x, style)
            else:
                x = self.norm_ff(x)

        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            if self.use_conditional_normalize:
                x = self.norm_ff(x, style)
            else:
                x = self.norm_ff(x)

        if self.conv_module is not None:
            if self.use_conditional_normalize:
                x = self.norm_final(x, style)
            else:
                x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask, style

        return x, mask, style
