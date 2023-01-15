from pathlib import Path
import os
from pyannote.audio import Model
from pyannote.database import get_protocol
from pyannote.audio.tasks import Segmentation
from copy import deepcopy
import pytorch_lightning as pl
from utils import test, create_train_list, csv_to_rttm
from pyannote.database import FileFinder
from pyannote.database.util import load_rttm
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.audio import Pipeline
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.core import Timeline, Segment


input_path = 'data/'
max_epochs = 500


data_files = list(Path(input_path).glob('*.csv*'))
rttm_file = 'train.rttm'
uem_file = 'train.uem'
lst_file = 'train.lst'

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
hparams = pipeline.parameters(instantiated=True)

res = []
for file_name in data_files:

    files_name_for_train = data_files[:]
    files_name_for_train.remove(file_name)

    # data preparation
    create_train_list(rttm_file, uem_file, lst_file, files_name_for_train)
    wav_file = str(file_name).replace('.csv', '.wav')
    gt_path = str(file_name).replace('.csv', '.rttm')

    if not os.path.exists(gt_path):
        csv_to_rttm(str(file_name), gt_path)

    diarization = pipeline(wav_file, num_speakers=2)

    metric = DiarizationErrorRate()

    _, groundtruth = load_rttm(gt_path).popitem()
    der = metric(groundtruth, diarization, uem=Timeline([Segment(120, 1000000000)]))
    print(f'{wav_file} pretrained diarization error rate = {100 * der:.1f}%')

    # train
    pretrained = Model.from_pretrained("pyannote/segmentation", use_auth_token='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')

    protocol = get_protocol('MyDatabase.Protocol.MyProtocol', preprocessors={"audio": FileFinder()})
    # for resource in protocol.train():
    #     print(resource["uri"])

    seg_task = Segmentation(protocol, duration=pretrained.specifications.duration, max_num_speakers=2)
    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task
    finetuned.setup(stage="fit")

    # finetuned.freeze_up_to('linear.0')
    # finetuned.freeze_up_to('linear.1')
    finetuned.freeze_up_to('lstm')

    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)
    trainer.fit(finetuned)
    finetuned.setup(stage="infer")

    pipeline_ft = SpeakerDiarization(
        segmentation=finetuned,
        embedding=pipeline.embedding,
        embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
        clustering=pipeline.klustering,
    )

    pipeline_ft.instantiate(hparams)

    diarization_ft = pipeline_ft(wav_file, num_speakers=2)

    metric = DiarizationErrorRate()
    der_ft = metric(groundtruth, diarization_ft, uem=Timeline([Segment(120, 1000000000)]))
    print(f'diarization error rate = {100 * der_ft:.1f}%')

    res.append((wav_file, der, der_ft))

    print()

for el in res:
    print(el)
