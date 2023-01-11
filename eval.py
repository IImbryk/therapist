import numpy as np

from utils import *
from tqdm import tqdm
from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection
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
parser.add_argument('--input_path', default='data_test/', type=str)


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

# model loading for for overlap!!!!
overlapped_hyperparameters = {"onset": 0.5, "offset": 0.5, "min_duration_on": 0.1, "min_duration_off": 0.1}

model = Model.from_pretrained("pyannote/segmentation", use_auth_token=True)
overlapp_model = OverlappedSpeechDetection(segmentation=model)
overlapp_model.instantiate(overlapped_hyperparameters)


file_metrics_path = Path('metrics_result.txt')
file_metrics = open(file_metrics_path, 'w')


# inference
print(f'{len(wav_list)} files to process')
der_result = []
for wav_file in wav_list:
    fname = wav_file.stem

    csv_files = f'{args.input_path}{fname}.csv'
    gt_path = f'{args.input_path}{fname}.rttm'

    diarization_path = f'output/{fname}_diarization.rttm'
    # overlap_path = f'output/{fname}_overlapped.rttm'
    # gt_overlap_path = f'{args.input_path}{fname}_gt_overlapped.rttm'
    # after_embedding_path = f'output/{fname}_after_embedding.rttm'
    figure_path = f'output/{fname}_plot_result'

    if not os.path.exists(gt_path):
        csv_to_rttm(csv_files, gt_path)

    # create gt overlap annotation
    # overlap_gt = get_overlap_reference(gt_path)
    # save_annotation(overlap_gt, gt_overlap_path)

    # diarization
    diarization = pipeline(wav_file, num_speakers=2)
    save_annotation(diarization, diarization_path)

    # overlap
    # gt_overlap = get_overlap_reference(gt_path)

    # overlap = get_overlapped_from_model(wav_file, ovl_model, params_ovl['offset'], params_ovl['onset'])
    # overlap = overlapp_model(wav_file)

    # # save result overlap in rttm file
    # save_annotation(overlap, overlap_path)
    # save_annotation(gt_overlap, gt_overlap_path)

    gt = load_rttm(gt_path)
    key_name = list(gt.keys())[0]
    groundtruth = gt[key_name]

    # diarization evaluation
    _, _, der = eval_diarization(gt_path, diarization_path, file_metrics)
    der_result.append(der)


print(f'Diarization Error Rate: {100 * np.average(der_result):.1f}% \n')
file_metrics.close()
