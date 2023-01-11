from utils import *
from pyannote.audio import Pipeline
import os
from pyannote.audio.pipelines import SpeakerDiarization
import torch

custom_model = False  # Tune
read_hyperparameters = False  # Tune
add_split_on_chunks = True  # Tune
input_path = 'data_test/'  # Tune

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

check_files_name(input_path)
wav_list = list(Path(input_path).glob('*.wav*'))

# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)

hparams = pipeline.parameters(instantiated=True)

# Load model for diarization
if custom_model:
    finetuned_model = torch.load('models/models_segmentation.pt', map_location=device)

    pipeline = SpeakerDiarization(
        segmentation=finetuned_model,
        embedding=pipeline.embedding,
        embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
        clustering=pipeline.klustering,
    )
    pipeline.instantiate(hparams)


if os.path.exists('hyperparameters/config.yaml') and read_hyperparameters == True:
    print('Base Params:', hparams)
    pipeline.load_params(Path('hyperparameters/config.yaml'))
    new_hyperparameters = pipeline.parameters(instantiated=True)
    print('New Params:', new_hyperparameters)


for wav_file in wav_list:
    fname = wav_file.stem

    diarization_path = f'output/{fname}_diarization.rttm'
    overlap_path = f'output/{fname}_overlapped.rttm'

    # diarization
    diarization = pipeline(wav_file, num_speakers=2)
    save_annotation(diarization, diarization_path)


    # save audio
    fs_wav, audio = wavfile.read(f'data_test/{fname}.wav')
    chunks = get_chunks(f'output/{fname}_diarization.rttm', fs_wav)
    if add_split_on_chunks:
        chunks = merge_chunk(chunks)
    save_audio(chunks,  fname, fs_wav, audio)


