import torch


def pad_sequences(batch_seq):
    batch_size = len(batch_seq)
    max_len = max(seq.size(0) for seq in batch_seq)
    channel_size = batch_seq[0].size(-1)

    padded_sequence = torch.zeros(batch_size, max_len, channel_size).to(batch_seq[0].device)

    for i, seq in enumerate(batch_seq):
        seq_len = seq.size(0)
        padded_sequence[i, :seq_len] = seq

    return padded_sequence

def average_by_duration(rep, duration):
    """
    프레임 단위의 representation을 phoneme 단위로 평균
    Args:
        rep: (frame_length, channel) 형태의 representation
        duration: phoneme별 프레임 수 리스트
    Returns:
        (phoneme_length, channel) 형태의 averaged representation
    """
    # rep = rep.transpose(0,1)
    current_pos = 0
    averaged_rep = []
    
    for d in duration:
        d = int(d)
        if d > 0:
            # 현재 phoneme에 해당하는 프레임들의 평균
            phone_rep = rep[current_pos:current_pos + d].mean(dim=0)
            averaged_rep.append(phone_rep)
            current_pos += d
    
    return torch.stack(averaged_rep)  # (phoneme_length, channel)

# if is_inference:
#     # Inference time: 저장된 representation 사용
#     intensity_rep = []
#     for emotion, duration in zip(emotions, ds):
#         emotion_id = emotion.item()
#         rep = self.intensity_info[emotion_id][intensity_level].to(hs.device)
#         # rep을 duration 길이에 맞게 확장
#         rep_expanded = rep.unsqueeze(0).expand(len(duration), -1)
#         intensity_rep.append(rep_expanded)
#     intensity_rep = torch.stack(intensity_rep)
# else:
#     # Training time
#     intensity_input = torch.cat([pfs.unsqueeze(-1), efs, ys], dim=2)
#     intensity_rep = []
#     for i, (emotion, duration) in enumerate(zip(emotions, ds)):
#         emotion_id = emotion.item()
#         if emotion_id == 0:  # Neutral
#             rep = torch.zeros(len(duration), 256).to(hs.device)
#         else:
#             rep, _ = self.intensity_model.intensity_extractor(
#                 pre_intensity_feature[i:i + 1],
#                 emotion.view(-1).unsqueeze(-1)
#             )
#             # duration 정보를 사용하여 phoneme 단위로 평균
#             rep = average_by_duration(rep, duration)
#         intensity_rep.append(rep)
#     intensity_rep = torch.stack(intensity_rep)  # (batch_size, phoneme_length, channel)