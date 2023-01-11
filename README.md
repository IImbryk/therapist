# Therapist


## Table of contents
* [Structure](#Structure)
* [Installation](#Installation)
* [Start](#Start)

## Structure

scripts:
* `train.py` -  data preparing for training and segmentation's models training. The result writes in \models.
Set max_epochs in code for count epoch optimization
* `eval.py` - model testing. The result writes in \output and metrics_result.txt
(Set custom_model and read_hyperparameters if you want to use custom's models and new hyperparameters)
* `infer.py` - model using. The result writes in \output
(Set custom_model and read_hyperparameters if you want to use custom's models and new hyperparameters)

* `tune_hyperparameters.py` - the pipeline hyper-parameters optimizing -- segmentation.threshold and clustering.threshold
(Set iter_count for count epoch optimization)


directoires:
* `hyperparameters` - hyperparameters (result of `tune_hyperparameters.py`, using in `infer.py`)
* `models` -  customn models (result of `train.py`)
* `data_train` - .wav and .csv for models training
* `data_test` - .wav and .csv for models testing
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
To get a result on new data type in terminal:

```
$ python infer.py --trained_model --read_hyperparameters --no_merge --input_path <DATA_PATH>
```

Meaning of the flags:
* `--trained_model` -- to use the trained model, default model from the library
* `--read_hyperparameters` -- to use optimized hyperparameters, without 
* `--no_merge` -- turn off merging audio from same speaker talk continuously
* `--input_path` -- path to data folder, default `\data`

All flags are optional. Can be typing: `$ python infer.py` -- it will be with models from pyanote, with default hyperparams and result merges by speaker.
