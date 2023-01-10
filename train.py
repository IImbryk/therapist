from pathlib import Path
import os
from utils import create_train_list
from pyannote.audio import Model
from pyannote.database import get_protocol
from pyannote.audio.tasks import Segmentation
from copy import deepcopy
import pytorch_lightning as pl
import time
import torch
from utils import test
from pyannote.database import FileFinder


max_epochs = 1  # Tune

path = os.path.dirname(os.path.abspath("__file__"))
input_path = os.path.join(path, 'data_train')
data_files = list(Path(input_path).glob('*.csv*'))

rttm_file = 'train.rttm'
uem_file = 'train.uem'
lst_file = 'train.lst'


create_train_list(rttm_file, uem_file, lst_file, data_files)


preprocessors = {"audio": FileFinder()}

protocol = get_protocol('MyDatabase.Protocol.MyProtocol', preprocessors=preprocessors)
for resource in protocol.train():
    print(resource["uri"])


pretrained = Model.from_pretrained("pyannote/segmentation", use_auth_token='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')

seg_task = Segmentation(protocol, duration=1, max_num_speakers=2)

finetuned = deepcopy(pretrained)
finetuned.task = seg_task
#

der_pretrained = test(model=pretrained, protocol=protocol, subset="train")
print(f"Local DER (pretrained) = {der_pretrained * 100:.1f}%")
start_time = time.time()
trainer = pl.Trainer(gpus=1, max_epochs=max_epochs)
trainer.fit(finetuned)
print("--- %s seconds ---" % (time.time() - start_time))


der_finetuned = test(model=finetuned, protocol=protocol, subset="train")
print(f"Local DER (finetuned) = {der_finetuned * 100:.1f}%")


# save model
torch.save(finetuned, 'models/models.pt')


