#!/usr/bin/env python3

"""Script to run the inference of text-to-speeech model."""
import os
import json
import argparse
import logging
import torch.quantization as quantization

from g2pk import G2p as ko_g2p
from g2p_en import G2p as en_g2p
from jamo import h2j

from distutils.version import LooseVersion
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Optional
from typing import Union
from text import text_to_sequence, phone_to_index
from Numbers2Words import num2text

import re
import numpy as np
import torch

from utils.model_wJCU import get_model, resume
from tts.utils.duration_calculator import DurationCalculator
from torch_utils.device_funcs import to_device
from torch_utils.set_all_random_seed import set_all_random_seed
from tts.fastspeech2.fastspeech2_JCU import FastSpeech2
from utils import config_argparse
from utils.types import str2bool
from utils.types import str2triple_str
from utils.types import str_or_none


class Text2Speech:
    """Text2Speech class.

    Examples:
        >>> from bin.tts_inference import Text2Speech
        >>> # Case 1: Load the local model and use Griffin-Lim vocoder
        >>> text2speech = Text2Speech(
        >>>     train_config="/path/to/config.yml",
        >>>     model_file="/path/to/model.pth",
        >>> )
        >>> # Case 2: Load the local model and the pretrained vocoder
        >>> text2speech = Text2Speech.from_pretrained(
        >>>     train_config="/path/to/config.yml",
        >>>     model_file="/path/to/model.pth",
        >>>     vocoder_tag="kan-bayashi/ljspeech_tacotron2",
        >>> )
        >>> # Case 3: Load the pretrained model and use Griffin-Lim vocoder
        >>> text2speech = Text2Speech.from_pretrained(
        >>>     model_tag="kan-bayashi/ljspeech_tacotron2",
        >>> )
        >>> # Case 4: Load the pretrained model and the pretrained vocoder
        >>> text2speech = Text2Speech.from_pretrained(
        >>>     model_tag="kan-bayashi/ljspeech_tacotron2",
        >>>     vocoder_tag="parallel_wavegan/ljspeech_parallel_wavegan.v1",
        >>> )
        >>> # Run inference and save as wav file
        >>> import soundfile as sf
        >>> wav = text2speech("Hello, World")["wav"]
        >>> sf.write("out.wav", wav.numpy(), text2speech.fs, "PCM_16")

    """

    def __init__(
        self,
        train_configs: Union[Path, str] = None,
        model_dir: Union[Path, str] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 10.0,
        use_teacher_forcing: bool = False,
        use_att_constraint: bool = False,
        backward_window: int = 1,
        forward_window: int = 3,
        speed_control_alpha: float = 1.0,
        vocoder_config: Union[Path, str] = None,
        vocoder_file: Union[Path, str] = None,
        dtype: str = "float32",
        device: str = "cpu",
        seed: int = 777,
        always_fix_seed: bool = False,
        symbol_path: str = '',
        dict_unit: str = 'phone',
        w_quantize: bool = False,
    ):
        """Initialize Text2Speech module."""

        (preprocess_config, _) = train_configs

        # setup model
        model = get_model(train_configs, device=device)
        if w_quantize:
            model = quantization.quantize_dynamic(model, dtype=torch.qint8)
            model_paths = model_dir.split('/')
            model_paths[-1] = 'quantized_' + model_paths[-1]
            model_dir = '/'.join(model_paths)
        model = resume(model_dir, model, device=device)
        model.to(dtype=getattr(torch, dtype)).eval()
        self.cleaner = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.device = device
        self.dtype = dtype
        self.model = model
        self.duration_calculator = DurationCalculator()
        self.use_teacher_forcing = use_teacher_forcing
        self.seed = seed
        self.always_fix_seed = always_fix_seed
        self.g2p_ko = ko_g2p()
        self.g2p_en = en_g2p()
        self.dict_unit = dict_unit
        self.n2w = num2text()

        # self.symbol_path = preprocess_config["path"]["symbol_path"]
        self.symbol_path = symbol_path
        with open(self.symbol_path, 'r') as symbol_file:
            self.symbols = json.load(symbol_file)
        # setup decoding config
        decode_conf = {}
        decode_conf.update(use_teacher_forcing=use_teacher_forcing)
        decode_conf.update(alpha=speed_control_alpha)

        self.decode_conf = decode_conf

    @torch.no_grad()
    def __call__(
        self,
        text: Union[str, torch.Tensor, np.ndarray],
        speech: Union[torch.Tensor, np.ndarray] = None,
        pitch: Union[torch.Tensor, np.ndarray] = None,
        durations: Union[torch.Tensor, np.ndarray] = None,
        energy: Union[torch.Tensor, np.ndarray] = None,
        spembs: Union[torch.Tensor, np.ndarray] = None,
        sids: Union[torch.Tensor, np.ndarray] = None,
        lids: Union[torch.Tensor, np.ndarray] = None,
        emotions: Union[torch.Tensor, np.ndarray] = None,
        intensity_level: Optional[str] = None,
        mel_length: Union[torch.Tensor, np.ndarray] = None,
        decode_conf: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run text-to-speech."""

        # prepare batch
        if isinstance(text, str):
            text = self.text_filter(text, lids)
            text = np.array(phone_to_index(text, self.symbols))
            # np.save(os.path.join('/home/matt/PycharmProjects/TTS_toolkit_multi_lingual_onnx/sample_text/', 'sample.npy'), text)
        batch = dict(text=text)
        if speech is not None:
            batch.update(feats=speech)
        if pitch is not None:
            batch.update(pitch=pitch)
        if durations is not None:
            batch.update(durations=durations)
        if energy is not None:
            batch.update(energy=energy)
        if spembs is not None:
            batch.update(spembs=spembs)
        if sids is not None:
            batch.update(sids=sids)
        if lids is not None:
            batch.update(lids=lids)
        if emotions is not None:
            batch.update(emotions=emotions)
        if intensity_level is not None:
            batch.update(intensity_level=intensity_level)
        if mel_length is not None:
            batch.update(mel_length=mel_length)
        batch = to_device(batch, self.device)

        # overwrite the decode configs if provided
        cfg = self.decode_conf
        if decode_conf is not None:
            cfg = self.decode_conf.copy()
            cfg.update(decode_conf)

        # inference
        if self.always_fix_seed:
            set_all_random_seed(self.seed)

        output_dict = self.model.inference(**batch, **cfg)

        # calculate additional metrics
        if output_dict.get("att_w") is not None:
            duration, focus_rate = self.duration_calculator(output_dict["att_w"])
            output_dict.update(duration=duration, focus_rate=focus_rate)

        return output_dict

    def text_filter(self, text, lids):
        # text = re.sub(r"([~/*\":―·ㆍ`!?°“”’‘≪≫〈〉<>（）「」《》{}|=\';.\(\)\[\]\-\s+])", " ", text)
        if lids == 0:
            text = re.sub(r"([/*\":―·ㆍ`!?°“”’‘≪≫〈〉<>（）「」《》{}|=\'\(\)\[\]\s+])", " ", text)
            text = text.replace('π', '파이').replace('花', '화').replace('華', '').replace('萬古風霜', '만고풍상').replace('處士',
                                                                                                                    '처사').replace(
                '非', '비').replace('%', '퍼센트').replace('ㅘ', '와').replace('㎐', '헤르츠').replace('℃', '도씨').replace('㎢',

                                                                                                        '제곱킬로미터')
            texts = text.split(' ')
            edited_text = list()

            if self.dict_unit == 'phone':
                for text in texts:
                    edited_text.append(self.g2p_ko(text))
                text = "".join(edited_text)
                text = h2j(text)
                temp_text = []
                for phone in text:
                    temp_text.append(phone)
                text = "{" + " ".join(temp_text) + "}"
                text = text.replace('-', 'd1')
                text = text.replace(';', 'd2')
                text = text.replace(',', 'd3')
                text = text.replace('.', 'd4')
                text = text.replace('~', 'd5')

            elif self.dict_unit == 'char':
                for text in texts:
                    edited_text.append(self.g2p_ko(text))
                text = "".join(edited_text)
                text = "{" + " ".join(text) + "}"
                text = text.replace(',', '')
        elif lids == 1:
            text = self.n2w.ConvertEng(text)
            text = re.sub("\&", " and ", text)
            text = re.sub(r"([/*\":\(\)\[\]\?\!\s+])", " ", text.upper())
            text = self.g2p_en(text)
            index = np.where(np.array(text) == ' ')[0]
            temp_text = []
            for i in range(len(text)):
                if not i in index:
                    temp_text.append(text[i])
            temp_text.append('')

            text = "{" + " ".join(temp_text) + "}"
            text = text.replace('-', 'd1')
            text = text.replace(',', 'd2')
            text = text.replace('.', 'd3')
            text = text.replace(';', 'd4')
            text = text.replace('~', 'd5')

            return text

        elif lids == 2:
            # text = re.sub(r"([/*\":\(\)\[\]\?\!\s\¿+])", " ", text)
            # g2p_generator = PyniniGenerator(word_list=[text], g2p_model_path='spanish_latin_america_mfa')
            # text = g2p_generator.generate_pronunciations()
            # text = list(text.values())[0][0]

            text = "{" + " ".join(text) + "}"
            text = text.replace('-', 'd1')
            text = text.replace(',', 'd2')
            text = text.replace('.', 'd3')
            text = text.replace(';', 'd4')
            text = text.replace('~', 'd5')

        return text

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None,
        vocoder_tag: Optional[str] = None,
        **kwargs: Optional[Any],
    ):
        """Build Text2Speech instance from the pretrained model.

        Args:
            model_tag (Optional[str]): Model tag of the pretrained models.
                Currently, the tags of espnet_model_zoo are supported.
            vocoder_tag (Optional[str]): Vocoder tag of the pretrained vocoders.
                Currently, the tags of parallel_wavegan are supported, which should
                start with the prefix "parallel_wavegan/".

        Returns:
            Text2Speech: Text2Speech instance.

        """
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader

            except ImportError:
                logging.error(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
                raise
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))

        if vocoder_tag is not None:
            if vocoder_tag.startswith("parallel_wavegan/"):
                try:
                    from parallel_wavegan.utils import download_pretrained_model

                except ImportError:
                    logging.error(
                        "`parallel_wavegan` is not installed. "
                        "Please install via `pip install -U parallel_wavegan`."
                    )
                    raise

                from parallel_wavegan import __version__

                # NOTE(kan-bayashi): Filelock download is supported from 0.5.2
                assert LooseVersion(__version__) > LooseVersion("0.5.1"), (
                    "Please install the latest parallel_wavegan "
                    "via `pip install -U parallel_wavegan`."
                )
                vocoder_tag = vocoder_tag.replace("parallel_wavegan/", "")
                vocoder_file = download_pretrained_model(vocoder_tag)
                vocoder_config = Path(vocoder_file).parent / "config.yml"
                kwargs.update(vocoder_config=vocoder_config, vocoder_file=vocoder_file)

            else:
                raise ValueError(f"{vocoder_tag} is unsupported format.")

        return Text2Speech(**kwargs)


def get_parser():
    """Get argument parser."""
    parser = config_argparse.ArgumentParser(
        description="TTS inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Note(kamo): Use "_" instead of "-" as separator.
    # "-" is confusing if written in yaml.
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The path of output directory",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=0,
        help="The number of gpus. 0 indicates CPU mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float16", "float32", "float64"],
        help="Data type",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="The number of workers used for DataLoader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="The batch size for inference",
    )

    group = parser.add_argument_group("Input data related")
    group.add_argument(
        "--data_path_and_name_and_type",
        type=str2triple_str,
        required=True,
        action="append",
    )
    group.add_argument(
        "--key_file",
        type=str_or_none,
    )
    group.add_argument(
        "--allow_variable_data_keys",
        type=str2bool,
        default=False,
    )

    group = parser.add_argument_group("The model configuration related")
    group.add_argument(
        "--train_config",
        type=str,
        help="Training configuration file",
    )
    group.add_argument(
        "--model_file",
        type=str,
        help="Model parameter file",
    )
    group.add_argument(
        "--model_tag",
        type=str,
        help="Pretrained model tag. If specify this option, train_config and "
        "model_file will be overwritten",
    )

    group = parser.add_argument_group("Decoding related")
    group.add_argument(
        "--maxlenratio",
        type=float,
        default=10.0,
        help="Maximum length ratio in decoding",
    )
    group.add_argument(
        "--minlenratio",
        type=float,
        default=0.0,
        help="Minimum length ratio in decoding",
    )
    group.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold value in decoding",
    )
    group.add_argument(
        "--use_att_constraint",
        type=str2bool,
        default=False,
        help="Whether to use attention constraint",
    )
    group.add_argument(
        "--backward_window",
        type=int,
        default=1,
        help="Backward window value in attention constraint",
    )
    group.add_argument(
        "--forward_window",
        type=int,
        default=3,
        help="Forward window value in attention constraint",
    )
    group.add_argument(
        "--use_teacher_forcing",
        type=str2bool,
        default=False,
        help="Whether to use teacher forcing",
    )
    parser.add_argument(
        "--speed_control_alpha",
        type=float,
        default=1.0,
        help="Alpha in FastSpeech to change the speed of generated speech",
    )
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.667,
        help="Noise scale parameter for the flow in vits",
    )
    parser.add_argument(
        "--noise_scale_dur",
        type=float,
        default=0.8,
        help="Noise scale parameter for the stochastic duration predictor in vits",
    )
    group.add_argument(
        "--always_fix_seed",
        type=str2bool,
        default=False,
        help="Whether to always fix seed",
    )

    group = parser.add_argument_group("Vocoder related")
    group.add_argument(
        "--vocoder_config",
        type=str_or_none,
        help="Vocoder configuration file",
    )
    group.add_argument(
        "--vocoder_file",
        type=str_or_none,
        help="Vocoder parameter file",
    )
    group.add_argument(
        "--vocoder_tag",
        type=str,
        help="Pretrained vocoder tag. If specify this option, vocoder_config and "
        "vocoder_file will be overwritten",
    )
    return parser

#
# def main(cmd=None):
#     """Run TTS model inference."""
#     print(get_commandline_args(), file=sys.stderr)
#     parser = get_parser()
#     args = parser.parse_args(cmd)
#     kwargs = vars(args)
#     kwargs.pop("config", None)
#     inference(**kwargs)
#
#
# if __name__ == "__main__":
#     main()