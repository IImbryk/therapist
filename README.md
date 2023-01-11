# Therapist


## Table of contents
* [Structure](#Structure)
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

	
## Start
To run this training type in terminal:

```
$ python3 train.py
```


1. Register on https://huggingface.co
2. Sign the User Agreements https://huggingface.co/pyannote/speaker-diarization
3. Go to https://huggingface.co/settings/tokens (User Access Tokens) and generate token and copy it

save token for linux:
```
python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('MY_HUGGINGFACE_TOKEN_HERE')"
```