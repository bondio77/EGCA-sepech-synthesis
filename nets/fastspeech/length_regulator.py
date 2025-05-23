#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Length regulator related modules."""

import logging
import torch
import torch.nn as nn

from nets.nets_utils import pad_list
from nets.nets_utils import pad


class LengthRegulator(nn.Module):
    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, alpha, max_len=None):
        output = list()
        mel_len = list()

        if alpha != 1.0:
            assert alpha > 0
            duration = torch.round(duration.float() * alpha).long()

        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_len is not None and not isinstance(max_len, float):
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(mel_len).to(x.device)

    def expand(self, batch, predicted):
        out = list()
        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, alpha, max_len=None):
        output, mel_len = self.LR(x, duration, alpha, max_len)
        return output, mel_len



# class LengthRegulator(nn.Module):
#     """Length regulator module for feed-forward Transformer.
#
#     This is a module of length regulator described in
#     `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
#     The length regulator expands char or
#     phoneme-level embedding features to frame-level by repeating each
#     feature based on the corresponding predicted durations.
#
#     .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
#         https://arxiv.org/pdf/1905.09263.pdf
#
#     """
#
#     def __init__(self, pad_value=0.0):
#         """Initilize length regulator module.
#
#         Args:
#             pad_value (float, optional): Value used for padding.
#
#         """
#         super().__init__()
#         self.pad_value = pad_value
#
#     def forward(self, xs, ds, alpha=1.0):
#         """Calculate forward propagation.
#
#         Args:
#             xs (Tensor): Batch of sequences of char or phoneme embeddings (B, Tmax, D).
#             ds (LongTensor): Batch of durations of each frame (B, T).
#             alpha (float, optional): Alpha value to control speed of speech.
#
#         Returns:
#             Tensor: replicated input tensor based on durations (B, T*, D).
#
#         """
#         if alpha != 1.0:
#             assert alpha > 0
#             ds = torch.round(ds.float() * alpha).long()
#
#         if ds.sum() == 0:
#             logging.warning(
#                 "predicted durations includes all 0 sequences. "
#                 "fill the first element with 1."
#             )
#             # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
#             #   It will be happened in inference with a bad duration predictor.
#             #   So we do not need to care the padded sequence case here.
#             ds[ds.sum(dim=1).eq(0)] = 1
#
#         repeat = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
#         return pad_list(repeat, self.pad_value)
