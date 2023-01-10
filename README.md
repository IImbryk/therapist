# therapist


## Table of contents
* [Structure](#Structure)
* [Start](#Start)

## Structure

scripts:
* `train.py` -  data preparing for training and models training. The result writes in \models.
Set max_epochs in code for count epoch optimization
* `infer.py` - model testing. The result writes in \output and metrics_result.txt
(Set custom_model and read_hyperparameters if you want to use custom's models and new hyperparameters)
* `tune_hyperparameters.py` - the pipeline hyper-parameters optimizing
(Set iter_count for count epoch optimization)


directoires:
* `hyperparameters` - hyperparameters (result of `tune_hyperparameters.py`, using in `infer.py`)
* `models` -  customn models (result of `train.py`)
* `data_train` - .wav and .csv for models training
* `data_test` - .wav and .csv for models testing
* `output` - results

	
## Start
To run this training write in terminal:

```
$ python3 train.py
```


