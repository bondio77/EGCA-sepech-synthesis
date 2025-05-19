import os
import json
from os.path import join

if __name__ == '__main__':
    data_path = '/home/matt/DB/multi/data/'
    output_path = '/home/matt/DB/multi/preprocessed/'
    speakers = os.listdir(data_path)
    speakers.sort()

    speaker_dict = dict()

    for i, spk in enumerate(speakers):
        speaker_dict[spk] = i

    with open(join(output_path, 'speakers.json'), 'w') as f:
        f.write(json.dumps(speaker_dict))
