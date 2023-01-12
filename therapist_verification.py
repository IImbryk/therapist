from pathlib import Path
from scipy.spatial.distance import cdist
from pyannote.audio import Model
from pyannote.audio import Inference
from utils import get_embedding, simular_speaker
from statistics import mode, mean
import random


class SpeakerVerification:
    def __init__(self, reference_path='reference_audio'):
        ref_list = list(Path(reference_path).glob('*.wav'))
        model = Model.from_pretrained("pyannote/embedding", use_auth_token=True)
        self.inference = Inference(model, window="whole")
        self.ref_embedding = get_embedding(ref_list, self.inference)

    def get_name_for_dir(self, dir_path):
        audio_list = random.choices(list(Path(dir_path).glob('*.wav')), k=3)
        chunk_embedding = get_embedding(audio_list, self.inference)

        name = []
        distance = []
        for emb in chunk_embedding.values():
            name_for_max, max_d = simular_speaker(emb, self.ref_embedding, cdist)
            name.append(name_for_max)
            distance.append(max_d)
        return mode(name), mean(distance)


if __name__ == "__main__":
    filename = 'SHAHAF_AVIGAIL_AUDIO'
    dir_path = f'output/{filename}_split/SPEAKER_00'
    verification = SpeakerVerification()
    print(verification.get_name_for_dir(dir_path))