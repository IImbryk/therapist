from pathlib import Path
from pyannote.audio import Model
from pyannote.database import get_protocol
from pyannote.audio.tasks import Segmentation
from copy import deepcopy
import pytorch_lightning as pl
from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.database import FileFinder
from pyannote.audio import Pipeline
from utils import *

path = os.path.dirname(os.path.abspath("__file__"))
input_path = 'data_train/'
# input_path = os.path.join(path, 'data_train')
data_files = list(Path(input_path).glob('*.csv*'))
rttm_file = 'train.rttm'
uem_file = 'train.uem'
lst_file = 'train.lst'

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")
# pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=True)

hparams = pipeline.parameters(instantiated=True)

max_epochs = 100  # 10, 20, 50, 100


file_metrics_path = Path('metrics_result.txt')
file_metrics = open(file_metrics_path, 'w')

for file_name in data_files:
    # data preparation
    create_train_list(rttm_file, uem_file, lst_file, [file_name], duration=120)

    protocol = get_protocol('MyDatabase.Protocol.MyProtocol', preprocessors={"audio": FileFinder()})
    for resource in protocol.train():
        print(resource["uri"])

    pretrained = Model.from_pretrained("pyannote/segmentation", use_auth_token='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')
    seg_task = Segmentation(protocol, duration=1, max_num_speakers=2)

    finetuned = deepcopy(pretrained)
    finetuned.task = seg_task
    finetuned.setup(stage="fit")

    # finetuned.freeze_up_to('classifier')
    # finetuned.freeze_up_to('linear.1')
    # finetuned.freeze_up_to('linear.0')
    finetuned.freeze_up_to('linear')

    trainer = pl.Trainer(gpus=1, max_epochs=max_epochs);
    trainer.fit(finetuned);

    pipeline = SpeakerDiarization(
        segmentation=finetuned,
        embedding=pipeline.embedding,
        embedding_exclude_overlap=pipeline.embedding_exclude_overlap,
        clustering=pipeline.klustering,
    )
    pipeline.instantiate(hparams)

    fname = file_name.stem

    csv_files = f'{input_path}{fname}.csv'
    gt_path = f'{input_path}{fname}.rttm'

    diarization_path = f'output/{fname}_diarization.rttm'

    if not os.path.exists(gt_path):
        csv_to_rttm(csv_files, gt_path)


    # diarization
    diarization = pipeline(f'{input_path}{fname}.wav', num_speakers=2)
    save_annotation(diarization, diarization_path)

    gt = load_rttm(gt_path)
    key_name = list(gt.keys())[0]
    groundtruth = gt[key_name]

    # diarization evaluation
    _, _, der = eval_diarization(gt_path, diarization_path, file_metrics)


    print(f'Diarization Error Rate: {100 * der:.1f}% \n')

file_metrics.close()