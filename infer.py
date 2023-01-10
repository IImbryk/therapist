from pyannote.database.util import load_rttm
from pathlib import Path
from utils import save_fig, chunks, save_annotation, csv_to_rttm, get_duration, eval_overlapped, eval_diarization, check_files_name, get_overlap_reference, test
from pyannote.audio import Pipeline
from tqdm import tqdm
import os
from pyannote.audio import Model
from pyannote.audio.pipelines import OverlappedSpeechDetection
from pyannote.audio.pipelines import SpeakerDiarization
import torch



custom_model = False  # Tune
read_hyperparameters = True  # Tune


input_path = 'data_test/'
torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

path = os.path.dirname(os.path.abspath("__file__"))
print('Current_dir', path)

check_files_name(input_path)
wav_list = list(Path(input_path).glob('*.wav*'))

HYPER_PARAMETERS = {}

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
hparams = pipeline.parameters(instantiated=True)

# Load model diarization and overlap
if custom_model:
    finetuned_model = torch.load('models/models.pt')

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


HYPER_PARAMETERS = {
    # onset/offset activation thresholds
    "onset": 0.56, "offset": 0.56,
    "min_duration_on": 0.0,
    "min_duration_off": 0.0
}


model = Model.from_pretrained("pyannote/segmentation", use_auth_token='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')
overlapp_model = OverlappedSpeechDetection(segmentation=model)
overlapp_model.instantiate(HYPER_PARAMETERS)


file_metrics_path = Path('metrics_result.txt')
file_metrics = open(file_metrics_path, 'w')

for wav_file in wav_list:
    fname = wav_file.stem

    csv_files = f'{input_path}{fname}.csv'
    gt_path = f'{input_path}{fname}.rttm'

    diarization_path = f'output/{fname}_diarization.rttm'
    overlap_path = f'output/{fname}_overlapped.rttm'
    gt_overlap_path = f'{input_path}{fname}_gt_overlapped.rttm'
    after_embedding_path = f'output/{fname}_after_embedding.rttm'
    figure_path = f'output/{fname}_plot_result'

    if not os.path.exists(gt_path):
        csv_to_rttm(csv_files, gt_path)

    overlap_gt = get_overlap_reference(gt_path)

    # save gt overlap annotation
    save_annotation(overlap_gt, gt_overlap_path)

    # print(f'for audio {fname}')
    # print('__________________________________________')
    # print(f'calculation of diarization')
    FILE = {'uri': f'{fname}', 'audio': wav_file}
    diarization = pipeline(wav_file, num_speakers=2)
    save_annotation(diarization, diarization_path)
    # overlap
    # print('__________________________________________')
    # print(f'calculation of overlay')
    gt_overlap = get_overlap_reference(gt_path)

    # overlap = get_overlapped_from_model(wav_file, ovl_model, params_ovl['offset'], params_ovl['onset'])
    overlap = overlapp_model(wav_file)
    # save result in rttm file
    save_annotation(overlap, overlap_path)
    save_annotation(gt_overlap, gt_overlap_path)

    gt = load_rttm(gt_path)
    key_name = list(gt.keys())[0]
    groundtruth = gt[key_name]

    # diarization evaluation
    eval_diarization(gt_path, diarization_path, file_metrics)

    # overlap evaluation
    len_audio = get_duration(str(wav_file))
    eval_overlapped(gt_overlap_path, overlap_path, len_audio, file_metrics)

    # plot
    plot_duration = 30
    for start, stop in tqdm(chunks(get_duration(str(wav_file)), plot_duration)):
        save_fig(groundtruth, diarization, overlap, start, stop, figure_path, fname)

file_metrics.close()