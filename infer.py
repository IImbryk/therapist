import os
import torch
import argparse
from pathlib import Path
from scipy.io import wavfile
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from utils import check_files_name, save_annotation, get_chunks, merge_chunk, save_audio


parser = argparse.ArgumentParser(description='therapist diarizetion')

parser.add_argument('--trained_model', action='store_true')
parser.add_argument('--read_hyperparameters', action='store_true')
parser.add_argument('--no_merge', action='store_true')
parser.add_argument('--input_path', default='data/', type=str)


args = parser.parse_args()

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

check_files_name(args.input_path)
wav_list = list(Path(args.input_path).glob('*.wav*'))

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)

hparams = pipeline.parameters(instantiated=True)

# Load model for diarization
if args.trained_model:
    finetuned_model = torch.load('models/models_segmentation.pt', map_location=device)

    pipeline = SpeakerDiarization(
        segmentation=finetuned_model,
        embedding=pipeline.embedding,
        embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
        clustering=pipeline.klustering,
    )
    pipeline.instantiate(hparams)

# update hyperparameters
if os.path.exists('hyperparameters/config.yaml') and args.read_hyperparameters:
    print('Base Params:', hparams)
    pipeline.load_params(Path('hyperparameters/config.yaml'))
    new_hyperparameters = pipeline.parameters(instantiated=True)
    print('New Params:', new_hyperparameters)

# inference
print(f'{len(wav_list)} files to process')
for wav_file in wav_list:
    fname = wav_file.stem

    print('processing file', fname)

    diarization_path = f'output/{fname}_diarization.rttm'
    overlap_path = f'output/{fname}_overlapped.rttm'

    # diarization
    diarization = pipeline(wav_file, num_speakers=2)
    save_annotation(diarization, diarization_path)

    # save audio
    fs_wav, audio = wavfile.read(wav_file)
    chunks = get_chunks(f'output/{fname}_diarization.rttm', fs_wav)

    if not args.no_merge:
        chunks = merge_chunk(chunks)
    save_audio(chunks,  fname, fs_wav, audio)