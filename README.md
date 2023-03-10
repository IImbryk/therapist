# Therapist


## Table of contents
* [Structure](#Structure)
* [Installation](#Installation)
* [Start](#Start)

## Structure

scripts:
* `train.py` -  train segmentation model. The resulted model writes in folder \models.

* `eval.py` - model evaluation. The result writes in \output and metrics_result.txt
(Set trained_model and read_hyperparameters if you want to use trained model and new hyperparameters)

* `infer.py` - processing data. The result writes in folder \output.
(Set trained_model and read_hyperparameters if you want to use trained model and new hyperparameters)

* `tune_hyperparameters.py` - the pipeline hyper-parameters optimizing -- segmentation.threshold and clustering.threshold

directoires:
* `hyperparameters` - hyperparameters (result of `tune_hyperparameters.py`, using in `infer.py`)
* `models` -  trained model (result of `train.py`)
* `data_train` - put .wav and .csv for models training
* `data_test` - put .wav and .csv for models testing
* `reference_audio` - put .wav of the therapist's speech examples
* `output` - results


## Installation
1) Install Python 3.8+ (though it might work with Python 3.7)

2) Install libraries. There's two ways installation liberies:
* use `requirements.txt`:

Download code, [open terminal](https://www.groovypost.com/howto/open-command-window-terminal-window-specific-folder-windows-mac-linux/) in `\therapist` directory and type:
```
pip install -r requirements.txt
```
* manually:
```
pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
pip install pyannote.audio
pip install huggingface_hub
```
Primarily requirements are driven by the [pyannote](https://github.com/pyannote/pyannote-audio) library.


### Authorization

Official [pyannote.audio](https://github.com/pyannote/pyannote-audio) pipelines are open-source, but gated. It means that you have to first accept users conditions on their respective Huggingface page to access the pretrained weights and hyper-parameters.
1. Register on [HuggingFace](https://huggingface.co)
2. Visit [speaker-diarization page](https://huggingface.co/pyannote/speaker-diarization) and accept the terms
3. Visit [segmentation page](https://huggingface.co/pyannote/segmentation) and accept the terms
4. Go to [Seetings (User Access Tokens)](https://huggingface.co/settings/tokens), generate token and copy it
5. Insert token in `token.txt` file

You can use the same token on different PC.


## Start
To get a result on a new data type in terminal:

```
$ python infer.py --trained_model --read_hyperparameters --no_merge --input_path <DATA_PATH> --path_reference_audio <REFERENCE_PATH>
```

Meaning of the flags:
* `--trained_model` -- use the trained model
* `--read_hyperparameters` -- use optimized hyperparameter 
* `--no_merge` -- turn off merging audio from same speaker talk continuously
* `--input_path` -- path to data folder, default `\data`
* `--path_reference_audio` -- path to reference data folder of therapist's audios, default `\reference_audio`

All flags are optional. For example `$ python infer.py` -- default models from pyanote with default hyperparams and result merges by speaker.
