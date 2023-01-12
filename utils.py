from pyannote.audio import Inference
from scipy.io import wavfile
import os
import contextlib
import wave
from pyannote.database.util import load_rttm
from matplotlib import pyplot as plt
from pyannote.core import notebook
from pathlib import Path
from pyannote.core import Annotation, Segment
from scipy.io.wavfile import read as read_wav
from pydub import AudioSegment
import yaml
import datetime
import shutil

def string_to_seconds(s):
    s = s.split(':')
    s = list(map(int, s))
    res = s[0] * 60 * 60 + s[1] * 60 + s[2] * 1 + s[3] / 100
    return res


def round_nearest(x, a):
    return round(x / a) * a


def chunks(file_duration, plot_duration):
    for k in range(0, file_duration, plot_duration):
        start, stop = k, k + plot_duration
        stop = min(stop, file_duration)
        yield start, stop


def get_duration(wav_file):
    with contextlib.closing(wave.open(wav_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return int(duration)


def save_annotation(annotation, rttm_path):
    if not os.path.exists('output/'):
        os.mkdir('output')
    with open(rttm_path, 'w') as fout:
        annotation.write_rttm(fout)
    return print(f'Save {rttm_path}')


# _split
def audio_split(diarization_path, fname):
    fs_wav, audio = wavfile.read(f'{fname}.wav')

    with open(diarization_path, 'r') as fout_csv:
        for i, line in enumerate(fout_csv):
            line = line.strip().split(' ')
            if float(line[4]) > 1:
                start_time = float(line[3]) * fs_wav
                finish_time = start_time + float(line[4]) * fs_wav

                # print(start_time,"_", finish_time)
                chunk = audio[int(start_time):int(finish_time)]
                name = line[7]
                if not os.path.exists(f'{fname}_split/{name}/'):
                    os.makedirs(f'{fname}_split/{name}/')
                chunk_name = f'{fname}_split/{name}/{name}_{i}.wav'

                wavfile.write(chunk_name, fs_wav, chunk)


def eval_overlapped(gt_rttm_path, res_rttm_path, len_audio, file=False):
    """
        input (str):
                the path of rttm file groundtruth,
                the path of rttm file result from model

        output (pyannote.core.annotation.Annotation):
                the annotation file for groundtruth
                the annotation file for result from model

        The function reads annotation from rttm and prints metrics.
      """
    # print('Overlap evaluation: ')

    gt = load_rttm(gt_rttm_path)
    key_name = list(gt.keys())[0]
    gt = gt[key_name]

    # res = load_rttm(res_rttm_path)['<NA>']
    res = load_rttm(res_rttm_path)
    key_name = list(res.keys())[0]
    res = res[key_name]

    from pyannote.metrics.detection import DetectionPrecision, DetectionRecall, DetectionAccuracy
    DetectionPrecision = DetectionPrecision(collar=0.05)
    DetectionRecall = DetectionRecall(collar=0.05)
    DetectionAccuracy = DetectionAccuracy(collar=0.05)

    metric = DetectionPrecision(gt, res, detailed=True)
    # print('Precision: ', metric['detection precision'])
    dp = metric['detection precision']
    metric = DetectionRecall(gt, res, detailed=True)
    # print('Recall: ', metric['detection recall'])
    dr = metric['detection recall']

    # print('sum', dp + dr)
    # print(metric)
    # print('duration_overlap_res', res.get_timeline().duration())
    # print('len_audio', len_audio)

    # print(f'percentage of overlapping talks (result): {100*res.get_timeline().duration()/len_audio:.2f}%' )

    # print('duration_overlap_gt', gt.get_timeline().duration())
    # print(f'percentage of overlapping talks (gt): {100*gt.get_timeline().duration()/len_audio:.2f}%' )

    metric = DetectionAccuracy(gt, res)
    # print('Accuracy: ', metric)

    # if we want to write metrics to file
    if file:
        file.write('Overlap \n')
        metric = DetectionPrecision(gt, res, detailed=True)
        file.write(f'Precision: {metric["detection precision"]}\n')
        metric = DetectionRecall(gt, res, detailed=True)
        file.write(f'Recall: {metric["detection recall"]}\n')

        file.write(
            f'percentage of overlapping talks (result): {100 * res.get_timeline().duration() / len_audio:.2f}% \n')
        file.write(f'percentage of overlapping talks (gt): {100 * gt.get_timeline().duration() / len_audio:.2f}% \n')

        metric = DetectionAccuracy(gt, res)
        file.write(f'Accuracy {metric}\n')

    # print('____________________________')
    return res, gt


def eval_diarization(gt_rttm_path, res_rttm_path, file=False):
    """
      input (str):
              the path of rttm file groundtruth,
              the path of rttm file result from model

      output (pyannote.core.annotation.Annotation):
              the annotation file for groundtruth
              the annotation file for result from model

      The function reads annotation from rttm and prints metrics.
    """
    # print('Diarization evaluation: ')
    gt = load_rttm(gt_rttm_path)
    key_name = list(gt.keys())[0]
    gt = gt[key_name]

    res = load_rttm(res_rttm_path)
    key_name = list(res.keys())[0]
    res = res[key_name]

    from pyannote.metrics.detection import DetectionPrecision, DetectionRecall
    from pyannote.metrics.diarization import DiarizationErrorRate

    DetectionPrecision = DetectionPrecision(collar=0.05, skip_overlap=True)
    DetectionRecall = DetectionRecall(collar=0.05, skip_overlap=True)

    metric = DetectionPrecision(gt, res, detailed=True)
    # print('Detection Precision: ', metric['detection precision'])
    metric = DetectionRecall(gt, res, detailed=True)
    # print('Detection Recall: ', metric['detection recall'])

    metric = DiarizationErrorRate()
    # der = metric(gt, res)
    # print(f'diarization error rate: {100 * der:.1f}%')

    if file:
        file.write('Diarization \n')
        metric = DetectionPrecision(gt, res, detailed=True)
        file.write(f'Precision: {metric["detection precision"]}\n')
        metric = DetectionRecall(gt, res, detailed=True)
        file.write(f'Recall: {metric["detection recall"]}\n')

        metric = DiarizationErrorRate()
        der = metric(gt, res)
        file.write(f'Diarization Error Rate: {100 * der:.1f}% \n')

    # print('____________________________')
    return res, gt, der


def save_fig(groundtruth, diarization, overlap, start, end, results_dir, fname, axis_name='overlap'):
    # print(start, end)
    notebook.crop = Segment(start, end)

    nrows = 3

    fig, ax = plt.subplots(nrows=nrows, ncols=1)
    fig.set_figwidth(20)
    fig.set_figheight(nrows * 2)

    notebook.plot_annotation(diarization, ax=ax[0], time=True, legend=True)
    ax[0].text(notebook.crop.start + 0.5, 0.1, 'diarization', fontsize=14)

    notebook.plot_annotation(overlap, ax=ax[1], time=True, legend=False)
    ax[1].text(notebook.crop.start + 0.5, 0.1, axis_name, fontsize=14)

    if groundtruth:
        notebook.plot_annotation(groundtruth, ax=ax[2], time=True, legend=False)
        ax[2].text(notebook.crop.start + 0.5, 0.1, 'ground truth', fontsize=14)

    fname = Path(results_dir) / f'{start}_{end}.jpg'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    plt.savefig(fname)
    plt.close()


def csv_to_rttm(data_file, out_file):
    periods = []
    speaker1_in_process = False
    speaker2_in_process = False
    with open(data_file) as fin:
        header = fin.readline().split(',')

        speaker1_name = header[1]
        speaker2_name = header[2]

        for line in fin:
            line = line.strip().split(',')

            if not line[0]:
                continue

            current_time = line[0]
            speaker1 = line[1]
            speaker2 = line[2]

            if speaker1_in_process:
                if speaker1:
                    end_time_sp1 = current_time
                else:
                    periods.append(((start_time_sp1, end_time_sp1), speaker1_name))
                    speaker1_in_process = False
            else:
                if speaker1:
                    start_time_sp1 = current_time
                    speaker1_in_process = True

            if speaker2_in_process:
                if speaker2:
                    end_time_sp2 = current_time
                else:
                    periods.append(((start_time_sp2, end_time_sp2), speaker2_name))
                    speaker2_in_process = False
            else:
                if speaker2:
                    start_time_sp2 = current_time
                    speaker2_in_process = True

        # концовка
        if speaker1_in_process:
            periods.append(((start_time_sp1, end_time_sp1), speaker1_name))

        if speaker2_in_process:
            periods.append(((start_time_sp2, end_time_sp2), speaker2_name))

    filename = str(Path(data_file).stem)
    with open(out_file, 'w') as fout:
        for el in periods:
            time, speaker = el
            start_time, end_time = string_to_seconds(time[0]), string_to_seconds(time[1])
            duration = end_time - start_time

            line = f'SPEAKER {filename} 1 {start_time} {round(duration, 2)} <NA> <NA> {speaker} <NA> <NA>\n'
            fout.write(line)


def create_train_list(rttm_file, uem_file, lst_file, data_files):
    with open(rttm_file, 'w') as fout_rttm, open(uem_file, 'w') as fout_uem, open(lst_file, 'w') as fout_lst:

        priv_time = datetime.datetime.strptime("00:00:00:00", "%H:%M:%S:%f")
        for_print = 0
        for data_file in data_files:
            filename = Path(data_file).stem

            fout_lst.write(filename + '\n')
            print(data_file)
            wav_file = Path(data_file).with_suffix('.wav')
            audio = AudioSegment.from_file(wav_file)
            fout_uem.write(f'{filename} NA 0.0 {round(audio.duration_seconds, 1)}' + '\n')

            periods = []
            speaker1_in_process = False
            speaker2_in_process = False
            with open(data_file) as fin:
                header = fin.readline().split(',')

                speaker1_name = header[1]
                speaker2_name = header[2]

                for line in fin:
                    line = line.strip().split(',')

                    if not line[0]:
                        continue

                    current_time = line[0]
                    speaker1 = line[1]
                    speaker2 = line[2]

                    if speaker1_in_process:
                        if speaker1:
                            end_time_sp1 = current_time
                        else:
                            periods.append(((start_time_sp1, end_time_sp1), speaker1_name))
                            speaker1_in_process = False
                    else:
                        if speaker1:
                            start_time_sp1 = current_time
                            speaker1_in_process = True

                    if speaker2_in_process:
                        if speaker2:
                            end_time_sp2 = current_time
                        else:
                            periods.append(((start_time_sp2, end_time_sp2), speaker2_name))
                            speaker2_in_process = False
                    else:
                        if speaker2:
                            start_time_sp2 = current_time
                            speaker2_in_process = True

                # концовка
                if speaker1_in_process:
                    periods.append(((start_time_sp1, end_time_sp1), speaker1_name))

                if speaker2_in_process:
                    periods.append(((start_time_sp2, end_time_sp2), speaker2_name))

            filename = str(Path(data_file).stem)
            for el in periods:
                time, speaker = el
                start_time, end_time = string_to_seconds(time[0]), string_to_seconds(time[1])
                duration = end_time - start_time

                line = f'SPEAKER {filename} 1 {start_time} {round(duration, 2)} <NA> <NA> {speaker} <NA> <NA>\n'
                fout_rttm.write(line)
    # print(data_files)


def get_overlap_reference(annotation_file_path):
    annotation = dict()
    with open(annotation_file_path) as fin:
        for line in fin:
            line = line.strip().split()
            start_time = float(line[3])
            duration = float(line[4])
            speaker = line[7]
            end_time = round(start_time + duration, 3)

            if speaker not in annotation:
                annotation[speaker] = [(start_time, end_time)]
            else:
                annotation[speaker].append((start_time, end_time))

    res = Annotation()
    speaker1, speaker2 = list(annotation.keys())
    for segment1 in annotation[speaker1]:
        for segment2 in annotation[speaker2]:
            s1, e1 = segment1
            s2, e2 = segment2
            x = max(s1, s2)
            y = min(e1, e2)
            if x < y:
                res[Segment(x, y)] = 'overlapped'
    return res


def get_overlap_annotation(res):
    annotation = Annotation()
    for line in res:
        start_time, end_time = line
        annotation[Segment(start_time, end_time)] = 'overlapped'

    return annotation


def change_sampling_rate(path_file, path_file_out):
    sampling_rate, data = read_wav(path_file)  # enter your filename
    print("Sampling rate before changing", sampling_rate)
    sound = AudioSegment.from_file(path_file, format='wav', frame_rate=44100)
    sound = sound.set_frame_rate(16000)
    sound.export(path_file_out, format='wav')
    sampling_rate, data = read_wav(path_file_out)
    print("Sampling after changing", sampling_rate)


def read_yml(file_path):
    with open(file_path, mode='r') as fp:
        params_main = yaml.load(fp, Loader=yaml.SafeLoader)
    return params_main


def write_yml(file_path, params):
    with open(file_path, mode='w') as f:
        yaml.dump(params, f)


def get_params_from_yml(file_path, params_name):
    params_result = {}
    params = read_yml(file_path)
    # print(params['params'])
    for i in params_name:
        params_result[i] = float(params['params'][i])
    return params_result


def replace_param(path_params_main, path_params_new_SAD, path_params_new_SCD):
    params_main = read_yml(path_params_main)
    params_new_SAD = read_yml(path_params_new_SAD)
    params_new_SCD = read_yml(path_params_new_SCD)

    params_main['params']['speech_turn_segmentation']['speech_activity_detection']['offset'] = params_new_SAD['params'][
        'offset']
    params_main['params']['speech_turn_segmentation']['speech_activity_detection']['onset'] = params_new_SAD['params'][
        'onset']
    params_main['params']['speech_turn_segmentation']['speaker_change_detection']['alpha'] = params_new_SCD['params'][
        'alpha']

    write_yml(path_params_main, params_main)

    # return params_main


def train_model(root_path, pretrained, Application, count_epochs):
    # print(root_path)
    print(pretrained)
    params = {}
    protocol = 'OWN.SpeakerDiarization.MixHeadset'
    params["subset"] = "train"

    warm_start = Path(pretrained)
    pretrained_config_yml = warm_start.parents[3] / "config.yml"
    pretrained_config_yml = Path(pretrained_config_yml)
    print(pretrained_config_yml)

    params["warm_start"] = warm_start

    params["epochs"] = count_epochs  # int(arg["--to"])

    root_dir = Path(root_path).expanduser().resolve(strict=True)
    app = Application(root_dir, training=True, pretrained_config_yml=pretrained_config_yml)
    app.train(protocol, **params)


def validation_model(path_train_dir, Application, start, end, every):
    train_dir = Path(path_train_dir).expanduser().resolve(strict=True)
    app = Application.from_train_dir(train_dir, training=False)
    params = {}

    params["subset"] = "train"

    params["start"] = start
    params["end"] = end
    params["every"] = every

    duration = getattr(app.task_, "duration", None)
    duration = float(duration)
    params["duration"] = duration
    metric = getattr(app.task_, "metric", None)
    params["metric"] = metric
    params["n_jobs"] = 1

    protocol = 'OWN.SpeakerDiarization.MixHeadset'

    app.validate(protocol, **params)


def train_validate_model(Application, new_model_dir_path, base_model_path, count_epochs=5, start=1, end=5, every=1):
    os.makedirs(new_model_dir_path, exist_ok=True)
    # print("!!",base_model_path)
    train_model(new_model_dir_path, base_model_path, Application, count_epochs)
    validation_model(f'{new_model_dir_path}/train/OWN.SpeakerDiarization.MixHeadset.train', Application, start, end,
                     every)


def check_files_name(path):
    for f in list(Path(path).glob('*.WAV*')):
        fname = f.stem
        old_file = os.path.join(path, f'{fname}.WAV')
        new_file = os.path.join(path, f'{fname}.wav')
        os.rename(old_file, new_file)


def test(model, protocol, subset="test"):
    from pyannote.audio.utils.signal import binarize
    from pyannote.audio.utils.metric import DiscreteDiarizationErrorRate
    from pyannote.audio.pipelines.utils import get_devices

    (device,) = get_devices(needs=1)
    metric = DiscreteDiarizationErrorRate()
    files = list(getattr(protocol, subset)())

    inference = Inference(model, device=device)

    for file in files:
        reference = file["annotation"]
        hypothesis = binarize(inference(file))
        uem = file["annotated"]
        _ = metric(reference, hypothesis, uem=uem)

    return abs(metric)


def get_chunks(diarization_path, fs_wav):
    with open(diarization_path, 'r') as fout_csv:
        chunks = []
        for i, line in enumerate(fout_csv):
            line = line.strip().split(' ')
            if float(line[4]) > 1:
                start_time = float(line[3]) * fs_wav
                finish_time = start_time + float(line[4]) * fs_wav
                # chunk = audio[int(start_time): int(finish_time)]
                name = line[7]
                chunks.append((start_time, finish_time, name))
    return chunks


def merge_chunk(chunks):
    cnt = 0
    chunks_result = []
    for start_time, finish_time, name in chunks:
        if cnt == 0:
            start = start_time
            end = finish_time
            prev_speaker = name
        else:
            if prev_speaker != name:
                chunks_result.append((start, end, prev_speaker))
                start = start_time
                end = finish_time
                prev_speaker = name
            else:
                end = finish_time
        cnt += 1
    chunks_result.append((start, end, prev_speaker))
    return chunks_result


def save_audio(chanks, fname, fs_wav, audio):
    # os.makedirs(f'output/{fname}_split', exist_ok=True)
    if os.path.exists(f'output/{fname}_split'):
        shutil.rmtree(f'output/{fname}_split')
    for i, col in enumerate(chanks):
        start_time, finish_time, name = col[0], col[1], col[2]
        chunk = audio[int(start_time): int(finish_time)]
        if not os.path.exists(f'output/{fname}_split/{name}/'):
            os.makedirs(f'output/{fname}_split/{name}/')
        chunk_name = f'output/{fname}_split/{name}/{name}_{i}.wav'

        wavfile.write(chunk_name, fs_wav, chunk)


def get_embedding(files_list, inference) -> dict:
    """
    get embedding for all audio from directory
    """
    embedding = {}
    for ref_file in files_list:
        file_name = Path(ref_file).stem
        embedding[file_name] = inference(ref_file)[None]
    return embedding


def simular_speaker(emb_audio, ref_embedding, cdist):
    name_for_max = None
    max_d = 0
    for ref_file, ref_emb in ref_embedding.items():
        d = cdist(emb_audio, ref_emb, metric='cosine')[0, 0]
        if d > max_d:
            max_d = d
            name_for_max = ref_file
    return name_for_max, max_d
