import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
# from emo_intensity.models import base_model


class EmotionalSpeechDataset(Dataset):
    def __init__(self, preprocessed_path, txt_paths, split='train'):
        self.preprocessed_path = preprocessed_path
        self.txt_paths = txt_paths
        self.split = split

        self.emotion_map = {
            "Neutral": 0, "Angry": 1, "Sad": 2, "Happy": 3, "Surprise": 4
        }

        self.samples = self.load_samples()
        self.pairs = self.create_pairs()

    def load_samples(self):
        samples = defaultdict(lambda: defaultdict(list))
        for speaker, txt_path in self.txt_paths.items():
            with open(txt_path, 'r') as f:
                for line in f:
                    filename, text, emotion = line.strip().split('\t')
                    samples[speaker][text].append((filename, emotion))
        return samples

    def create_pairs(self):
        pairs = []
        for speaker, texts in self.samples.items():
            for text, samples in texts.items():
                neutral_samples = [s for s in samples if s[1] == "Neutral"]
                other_samples = [s for s in samples if s[1] != "Neutral"]

                for neutral_sample in neutral_samples:
                    for other_sample in other_samples:
                        pairs.append((speaker, neutral_sample, other_sample, text))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def load_features(self, speaker, emotion, filename):
        mel = np.load(os.path.join(self.preprocessed_path, "mel", f"{speaker}-{emotion}-mel-{filename}.npy"))
        pitch = np.load(os.path.join(self.preprocessed_path, "pitch_per_frame",
                                     f"{speaker}-{emotion}-pitch_per_frame-{filename}.npy"))
        energy = np.load(
            os.path.join(self.preprocessed_path, "energy_frame", f"{speaker}-{emotion}-energy_frame-{filename}.npy"))

        min_length = min(mel.shape[1], pitch.shape[0], energy.shape[0])
        mel = mel[:, :min_length]
        pitch = pitch[:min_length, np.newaxis]
        energy = energy[:min_length, np.newaxis]

        x = np.concatenate([pitch, energy, mel.transpose()], axis=1)
        return torch.FloatTensor(x), min_length

    def __getitem__(self, idx):
        try:
            speaker, (filename_neu, emotion_neu), (filename_emo, emotion_emo), text = self.pairs[idx]

            x_emo, emo_length = self.load_features(speaker, emotion_emo, filename_emo)
            x_neu, neu_length = self.load_features(speaker, emotion_neu, filename_neu)

            emotion_class = torch.LongTensor([self.emotion_map[emotion_emo]])
            y_emo = torch.LongTensor([self.emotion_map[emotion_emo]])
            y_neu = torch.LongTensor([self.emotion_map[emotion_neu]])

            return x_emo, x_neu, emotion_class, y_emo, y_neu, emo_length, neu_length
        except:
            # print(self.pairs[idx])
            return None

def collate_fn(batch):
    try:
        x_emo, x_neu, emotion_class, y_emo, y_neu, emo_length, neu_length = zip(*batch)

        x_emo_padded = torch.nn.utils.rnn.pad_sequence(x_emo, batch_first=True)
        x_neu_padded = torch.nn.utils.rnn.pad_sequence(x_neu, batch_first=True)

        max_len = x_emo_padded.shape[1]
        lengths_emo = torch.tensor([l / max_len for l in emo_length], device=x_emo_padded.device)  # (batch,)
        lengths_neu = torch.tensor([l / max_len for l in neu_length], device=x_neu_padded.device)  # (batch,)

        return (
            x_emo_padded,
            x_neu_padded,
            torch.stack(emotion_class),
            torch.stack(y_emo),
            torch.stack(y_neu),
            lengths_emo,
            lengths_neu
        )
    except:
        return None


