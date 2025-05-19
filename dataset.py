import os
import math
import json
import random

import kaldiio

import numpy as np

from os.path import join
from text import text_to_sequence, phone_to_index
from torch.utils.data import Dataset
from utils.tools import pad_1D, pad_2D
from glob import glob

class Dataset(Dataset):
    def __init__(
        self, filename, preprocess_config, batch_size, sort=False, drop_last=False, use_teacher_forcing=False,
    ):
        self.dataset_name = preprocess_config["dataset"]
        self.preprocessed_path = preprocess_config["path"]["preprocessed_path"]
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.symbol_path = preprocess_config["path"]["symbol_path"]
        self.batch_size = batch_size

        self.basename, self.speaker, self.text, self.raw_text, self.emotion = self.process_meta(filename)
        self.use_teacher_forcing = use_teacher_forcing
        with open(self.symbol_path, 'r') as symbol_file:
            self.symbols = json.load(symbol_file)

        try:
            self.xvectors = self.load_xvectors(join(self.preprocessed_path, "spk_xvector.ark"))
        except:
            self.xvectors = None
        self.preprocessed_path
        with open(join(self.preprocessed_path, "speakers.json")) as f:
            self.speaker_map = json.load(f)

        self.sort = sort
        self.drop_last = drop_last
        self.emo_dict = {
            "Neutral":0,
            "Angry":1,
            "Sad":2,
            "Surprise":4,
            "Happy":3
        }


    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        emotion = self.emotion[idx]
        emotion_id = self.emo_dict[emotion]
        phone = np.array(phone_to_index(self.text[idx], self.symbols))

        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            f"{speaker}-{emotion}-mel-{basename}.npy",
        )
        mel = np.load(mel_path)
        mel = mel.T

        pitch_per_phoneme_path = os.path.join(
            self.preprocessed_path,
            "pitch_per_phoneme",
            f"{speaker}-{emotion}-pitch_per_phoneme-{basename}.npy"
        )
        pitch_per_phoneme = np.load(pitch_per_phoneme_path)

        forward = pitch_per_phoneme[:-1]
        # forward = np.insert(forward,0,0)
        pitch_diff = pitch_per_phoneme[1:] - forward
        pitch_diff = np.insert(pitch_diff,0,0)


        pitch_per_frame_path = os.path.join(
            self.preprocessed_path,
            "pitch_per_frame",
            f"{speaker}-{emotion}-pitch_per_frame-{basename}.npy"
        )
        pitch_per_frame = np.load(pitch_per_frame_path)

        pitch_max_per_phoneme_path = os.path.join(
            self.preprocessed_path,
            "pitch_max_per_phoneme",
            f"{speaker}-{emotion}-pitch_max_per_phoneme-{basename}.npy"
        )
        pitch_max_per_phoneme = np.load(pitch_max_per_phoneme_path)

        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            f"{speaker}-{emotion}-energy-{basename}.npy"
        )
        energy = np.load(energy_path)

        energy_frame_path = os.path.join(
            self.preprocessed_path,
            "energy_frame",
            f"{speaker}-{emotion}-energy_frame-{basename}.npy"
        )
        energy_frame = np.load(energy_frame_path)

        # mel_frame_path = os.path.join(
        #     self.preprocessed_path,
        #     "mel_frame",
        #     f"{speaker}-{emotion}-mel-{basename}.npy"
        # )
        # mel_frame = np.load(mel_frame_path)

        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            f"{speaker}-{emotion}-duration-{basename}.npy"
        )
        duration = np.load(duration_path)

        if self.xvectors is not None:
            xvector = self.xvectors[speaker]



        pitch_per_phoneme = np.pad(pitch_per_phoneme, (0, 1), 'constant', constant_values=0)
        pitch_per_frame = np.pad(pitch_per_frame, (0, 1), 'constant', constant_values=0)
        pitch_max_per_phoneme = np.pad(pitch_max_per_phoneme, (0, 1), 'constant', constant_values=0)
        # pitch_diff =  np.pad(pitch_diff, (0, 1), 'constant', constant_values=0)
        energy = np.pad(energy, (0, 1), 'constant', constant_values=0)
        energy_frame = np.pad(energy_frame, (0, 1), 'constant', constant_values=0)
        duration = np.pad(duration, (0, 1), 'constant', constant_values=1)
        mel = np.pad(mel, ((0,1), (0,0)), 'constant', constant_values=0)
        lang = 1

        # if speaker_id <= 74:
        #     lang = 0
        # elif speaker_id <= 150:
        #     lang = 1
        # else:
        #     lang = 2

        if self.xvectors is not None:
            sample = {
                "id": basename,
                "speaker": speaker_id,
                "emotion": emotion_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                # "mel_frame": mel_frame,
                "pitch_per_phoneme": pitch_per_phoneme,
                "pitch_per_frame": pitch_per_frame,
                "energy": energy,
                "energy_frame": energy_frame,
                "duration": duration,
                "xvector": xvector,
            }
        else:
            sample = {
                "id": basename,
                "speaker": speaker_id,
                "emotion": emotion_id,
                "text": phone,
                "raw_text": raw_text,
                "mel": mel,
                # "mel_frame": mel_frame,
                "pitch_per_phoneme": pitch_per_phoneme,
                "pitch_per_frame": pitch_per_frame,
                "pitch_max_per_phoneme": pitch_max_per_phoneme,
                "pitch_diff": pitch_diff,
                "energy": energy,
                "energy_frame": energy_frame,
                "duration": duration,
                "lang": lang
            }

        return sample



    def process_meta(self, filename):
        with open(
            os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            dataset = f.readlines()
            random.shuffle(dataset)
            for line in dataset:
                n, s, t, r, e = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                emotion.append(e)
            return name, speaker, text, raw_text, emotion

    # def process_meta(self, filename):
    #     with open(
    #             os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
    #     ) as f:
    #         name = []
    #         speaker = []
    #         text = []
    #         raw_text = []
    #         emotion = []
    #         dataset = f.readlines()
    #
    #         # Neutral 감정만 필터링
    #         neutral_dataset = [line for line in dataset if line.strip("\n").split("|")[4] == "Neutral"]
    #         random.shuffle(neutral_dataset)
    #
    #         for line in neutral_dataset:  # dataset 대신 neutral_dataset 사용
    #             n, s, t, r, e = line.strip("\n").split("|")
    #             name.append(n)
    #             speaker.append(s)
    #             text.append(t)
    #             raw_text.append(r)
    #             emotion.append(e)
    #         return name, speaker, text, raw_text, emotion

    def load_xvectors(self, xvector_path):
        xvectors = {k: np.array(v) for k, v in kaldiio.load_ark(xvector_path)}
        return xvectors

    def reprocess(self, data, idxs):
        ids = [data[idx]["id"] for idx in idxs]
        speakers = [data[idx]["speaker"] for idx in idxs]
        emotions = [data[idx]["emotion"] for idx in idxs]
        texts = [data[idx]["text"] for idx in idxs]
        raw_texts = [data[idx]["raw_text"] for idx in idxs]
        mels = [data[idx]["mel"] for idx in idxs]
        # mel_frame = [data[idx]["mel_frame"] for idx in idxs]
        pitches_per_phoneme = [data[idx]["pitch_per_phoneme"] for idx in idxs]
        pitches_per_frame = [data[idx]["pitch_per_frame"] for idx in idxs]
        pitches_max_per_phoneme = [data[idx]["pitch_max_per_phoneme"] for idx in idxs]
        pitch_diff = [data[idx]["pitch_diff"] for idx in idxs]



        energies = [data[idx]["energy"] for idx in idxs]
        energy_frames = [data[idx]["energy_frame"] for idx in idxs]
        durations = [data[idx]["duration"] for idx in idxs]

        langs = [data[idx]["lang"] for idx in idxs]

        if self.xvectors is not None:
            xvector = [data[idx]["xvector"] for idx in idxs]

        text_lens = np.array([text.shape[0] for text in texts])
        mel_lens = np.array([mel.shape[0] for mel in mels])

        pitch_per_phoneme_lens = np.array([pitch.shape[0] for pitch in pitches_per_phoneme])
        pitch_per_frame_lens = np.array([pitch.shape[0] for pitch in pitches_per_frame])
        pitch_max_per_phoneme_lens = np.array([pitch.shape[0] for pitch in pitches_max_per_phoneme])
        pitch_diff_lens = np.array([diff.shape[0] for diff in pitch_diff])
        energy_lens = np.array([energy.shape[0] for energy in energies])
        energy_frame_lens = np.array([energy_frame.shape[0] for energy_frame in energy_frames])
        duration_lens = np.array([duration.shape[0] for duration in durations])

        speakers = np.array(speakers)
        emotions = np.array(emotions)

        texts = pad_1D(texts)
        # mels =
        mels = pad_2D(mels)
        # mel_frame = pad_2D(mel_frame)
        pitches_per_phoneme = pad_1D(pitches_per_phoneme)
        pitches_per_frame = pad_1D(pitches_per_frame)
        pitches_max_per_phoneme = pad_1D(pitches_max_per_phoneme)
        pitch_diff = pad_1D(pitch_diff)
        energies = pad_1D(energies)
        energy_frames = pad_1D(energy_frames)
        durations = pad_1D(durations)

        langs = np.array(langs)

        if self.xvectors is not None:
            xvector = np.array(xvector)
            if self.use_teacher_forcing:
                return {
                    "base_name": ids,
                    # raw_texts,
                    # speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                    "spemb": xvector,
                }
            else:
                return {
                    # raw_texts,
                    # speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                    "spemb": xvector,
                }
        else:
            if self.use_teacher_forcing:
                return {
                    "base_name": ids,
                    # raw_texts,
                    "sids": speakers,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "energy": energies,
                    "energy_lengths": energy_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                    "lids": langs
                }
            else:
                return {
                    # raw_texts,
                    "sids": speakers,
                    "emotions": emotions,
                    "text": texts,
                    "text_lengths": text_lens,
                    # max(text_lens),
                    "feats": mels,
                    # "feats_frame": mel_frame,
                    "feats_lengths": mel_lens,
                    # max(mel_lens),
                    "pitch_per_phoneme": pitches_per_phoneme,
                    "pitch_per_phoneme_lengths": pitch_per_phoneme_lens,
                    "pitch_per_frame": pitches_per_frame,
                    "pitch_per_frame_lengths": pitch_per_frame_lens,
                    "pitch_max_per_phoneme": pitches_max_per_phoneme,
                    "pitch_max_per_phoneme_lengths": pitch_max_per_phoneme_lens,
                    "pitch_diff": pitch_diff,
                    "pitch_diff_lengths": pitch_diff_lens,
                    "energy": energies,
                    "energy_frame": energy_frames,
                    "energy_lengths": energy_lens,
                    "energy_frame_lengths": energy_frame_lens,
                    "durations": durations,
                    "durations_lengths": duration_lens,
                    "lids": langs
                }

    def collate_fn(self, data):
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        # idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        # output = list()
        # for idx in idx_arr:
            # output.append(self.reprocess(data, idx))
        output = self.reprocess(data, idx_arr)

        return output


class TextDataset(Dataset):
    def __init__(self, filepath, preprocess_config):
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

        self.basename, self.speaker, self.text, self.raw_text = self.process_meta(
            filepath
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            self.speaker_map = json.load(f)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        basename = self.basename[idx]
        speaker = self.speaker[idx]
        speaker_id = self.speaker_map[speaker]
        raw_text = self.raw_text[idx]
        phone = np.array(text_to_sequence(self.text[idx], self.cleaners))

        return (basename, speaker_id, phone, raw_text)

    def process_meta(self, filename):
        with open(
                os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
        ) as f:
            name = []
            speaker = []
            text = []
            raw_text = []
            emotion = []
            dataset = f.readlines()

            # Neutral 감정만 필터링
            neutral_dataset = [line for line in dataset if line.strip("\n").split("|")[4] == "Neutral"]
            random.shuffle(neutral_dataset)

            for line in neutral_dataset:  # dataset 대신 neutral_dataset 사용
                n, s, t, r, e = line.strip("\n").split("|")
                name.append(n)
                speaker.append(s)
                text.append(t)
                raw_text.append(r)
                emotion.append(e)
            return name, speaker, text, raw_text, emotion
    # def process_meta(self, filename):
    #     with open(filename, "r", encoding="utf-8") as f:
    #         name = []
    #         speaker = []
    #         text = []
    #         raw_text = []
    #         for line in f.readlines():
    #             n, s, t, r = line.strip("\n").split("|")
    #             name.append(n)
    #             speaker.append(s)
    #             text.append(t)
    #             raw_text.append(r)
    #         return name, speaker, text, raw_text

    def collate_fn(self, data):
        ids = [d[0] for d in data]
        speakers = np.array([d[1] for d in data])
        texts = [d[2] for d in data]
        raw_texts = [d[3] for d in data]
        text_lens = np.array([text.shape[0] for text in texts])

        texts = pad_1D(texts)

        return ids, raw_texts, speakers, texts, text_lens, max(text_lens)

