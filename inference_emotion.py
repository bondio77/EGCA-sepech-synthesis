import argparse
import os
import yaml
import torch
import json
import numpy as np
import soundfile as sf
from tqdm import tqdm
from bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model


def main(configs):
    emo_dict = {
        "Neutral": 0,
        "Angry": 1,
        "Sad": 2,
        "Surprise": 4,
        "Happy": 3
    }
    language_num = 1
    acoustic_dir = f"/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/model_checkpoint/context_EGCA_mamba1_8/1000epoch.pth"
    output_dir = f"/home/lee/Desktop/INSUNG/intensity_fs2/tts_tool/model_checkpoint/context_EGCA_mamba1_8/test1000"
    voco_dir = f"./esd_vocoder/generator_only.pth"

    symbol_path = ('/home/lee/Desktop/INSUNG/intensity_fs2/ESD_preprocess_vocs_1/symbols.json')

    text2speech = Text2Speech(
        train_configs=configs,
        model_dir=acoustic_dir,
        device='cuda:1',
        threshold=0.5,
        minlenratio=0.0,
        maxlenratio=10.0,
        backward_window=1,
        forward_window=3,
        symbol_path=symbol_path,
        dict_unit='phone',
        w_quantize=False,
    )

    preprocess_config, _ = configs
    vocoder = load_model(voco_dir).to('cuda:1').eval()

    intensity_level = ['min', 'med', 'max']

    with open('/home/lee/Desktop/INSUNG/intensity_fs2/ESD_preprocess_vocs_1/speakers.json', 'r') as speaker_file:
        speakers = json.load(speaker_file)

    inverted_speakers = {value: key for key, value in speakers.items()}
    # val_list = open('/home/user/PycharmProjects/emotion_inensity/intensity_fs2/ESD_preprocess_voc_fix/test.txt','r').readlines()
    val_list = open('../test.txt','r').readlines()
    # val_list = open('./emospeech_val.txt','r').readlines()
    # val_list = open('./train_weired.txt','r').readlines()

    # val_list = ['Monster made a deep bow.'  # 11
    #     , 'All smile were real and the happier，the more sincere.'  # 12
    #     , 'A divine wrath made her blue eyes awful.'  # 14
    #     , 'In which fox loses a tail and its elder sister finds one.'  # 15
    #     , 'Who is been repeating all that hard stuff to you?']  # 17'

    for j, i in tqdm(enumerate(val_list)):
        wav_name, spk, phone, txt, emotion = i.split('|')
        # spk, txt, emotion = i.split('|')
        emotion = emotion.replace('\n','')
        # if emotion[:-1] == 'Neutral':
        # id_list = ['0011', '0012', '0014', '0015', '0017']
        # if speakers[spk] < 75:
        #     continue
        for strength in intensity_level:
            with torch.no_grad():

                # if emotion != 'Angry':
                #     continue
                    # emo_id = emo_dict['Neutral']
                    # spk_id = speakers[id_list[j]]

                emo_id = emo_dict[emotion]
                spk_id = speakers[spk]
                output_dict = text2speech(txt, intensity_level=f'{strength}', emotions=torch.from_numpy(np.array(emo_id)).long().unsqueeze(0),sids=torch.from_numpy(np.array(spk_id)).long().unsqueeze(0), lids=torch.from_numpy(np.array(language_num)).long().unsqueeze(0))
                # output_dict = text2speech(i, intensity_level=f'{strength}', emotions=torch.from_numpy(np.array(emo_id)).long().unsqueeze(0),sids=torch.from_numpy(np.array(spk_id)).long().unsqueeze(0), lids=torch.from_numpy(np.array(language_num)).long().unsqueeze(0))


                # path = ['1_9.npz.npy', '2_4.npz.npy', '4_6.npz.npy', '2_5.npz.npy']
                # for pt in path:
                #     mel = np.load(f'/home/user/PycharmProjects/emotion_inensity/intensity_fs2/{pt}', allow_pickle=True)
                wav = vocoder.inference(output_dict["feat_gen"])
                    # wav = vocoder.inference(mel)

                # save_path = output_dir + '/' + id_list[j] + '/' + emotion[:-1] + '/' + strength +'/'+ i + '.wav'
                save_path = output_dir + '/' + spk + '/' + strength + '/' + emotion.replace('\n','') +'/'+ wav_name +'.wav'
                # save_path = output_dir + '/' + spk + '/' + strength + '/' + emotion.replace('\n','') + txt +'.wav'
                # save_path = output_dir + '/' f'{pt[:3]}.wav'

                txt_path = save_path.replace('wav','txt')
                if os.path.isdir(os.path.dirname(save_path)) == 0:
                    os.makedirs(os.path.dirname(save_path))

                f = open(txt_path,'w')
                f.write(i.replace('\n',''))
                f.close()
                sf.write(save_path, wav.cpu().numpy(), 24000)
            # else:
            #     continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--preprocess_config", type=str, default="./config/esd/preprocess.yaml", help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, default="./config/esd/model_6L.yaml", help="path to model.yaml"
    )
    args = parser.parse_args()

    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)

    configs = (preprocess_config, model_config)

    main(configs)




















 # spk_list = ['0011', '0012', '0013', '0014', '0015', '0016', '0017', '0018', '0019', '0020']
 #    # emotion = 'Neutral'
 #    for i in tqdm(val_list):
 #        emotion, txt = i.split('|')
 #        emotion = emotion[0].upper() + emotion[1:]
 #        for spk in spk_list:
 #            with torch.no_grad():
 #                emo_id = emo_dict[emotion]
 #                spk_id = speakers[spk]
 #                output_dict = text2speech(txt, intensity_level='med', emotions=torch.from_numpy(np.array(emo_id)).long().unsqueeze(0),sids=torch.from_numpy(np.array(spk_id)).long().unsqueeze(0), lids=torch.from_numpy(np.array(language_num)).long().unsqueeze(0))
 #                wav = vocoder.inference(output_dict["feat_gen"])
 #
 #                save_path = output_dir + '/' + spk + '/' + f'{emotion}' +'/' + f'{spk}_{emotion}_{i}.wav'
 #                txt_path = save_path.replace('wav','txt')
 #                if os.path.isdir(os.path.dirname(save_path)) == 0:
 #                    os.makedirs(os.path.dirname(save_path))
 #
 #                # f = open(txt_path,'w')
 #                # f.write(txt.replace('\n',''))
 #                # f.close()
 #                sf.write(save_path, wav.cpu().numpy(), 24000)