from pyannote.audio.pipelines import SpeakerDiarization
from pyannote.pipeline import Optimizer
from pyannote.audio import Model
from pyannote.database import get_protocol, FileFinder
from pyannote.audio import Pipeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pathlib import Path

iter_count = 2  # Tune

preprocessors = {"audio": FileFinder()}

dataset = get_protocol('MyDatabase.Protocol.MyProtocol', preprocessors=preprocessors)


pretrained_model = Model.from_pretrained("pyannote/segmentation", use_auth_token='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')
pretrained_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token='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')

pretrained_hyperparameters = pretrained_pipeline.parameters(instantiated=True)
print('Base Params:', pretrained_hyperparameters)

metric = DiarizationErrorRate()

for file in dataset.train():
    # apply pretrained pipeline
    file["pretrained pipeline"] = pretrained_pipeline(file)

    # evaluate its performance
    metric(file["annotation"], file["pretrained pipeline"], uem=file["annotated"])

print(f"The pretrained pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric):.1f}% on {dataset.name} test set.")


# optimizing segmentation.threshold
pipeline = SpeakerDiarization(
    segmentation=pretrained_model,
    clustering="OracleClustering",
)
# as reported in the technical report, min_duration_off can safely be set to 0.0
pipeline.freeze({"segmentation": {"min_duration_off": 0.0}})

optimizer = Optimizer(pipeline)
dev_set = list(dataset.train())

iterations = optimizer.tune_iter(dev_set, show_progress=False)
best_loss = 1.0
for i, iteration in enumerate(iterations):
    print(f"Best segmentation threshold so far: {iteration['params']['segmentation']['threshold']}")
    if i > iter_count:
        break

# the optimized value of segmentation.threshold
best_segmentation_threshold = optimizer.best_params["segmentation"]['threshold']

pipeline = SpeakerDiarization(
    segmentation=pretrained_model,
    embedding=pretrained_pipeline.embedding,
    embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
    clustering=pretrained_pipeline.klustering,
)

pipeline.freeze({
    "segmentation": {
        "threshold": best_segmentation_threshold,
        "min_duration_off": 0.0,
    },
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 15,
    },
})

# optimize clustering.threshold
optimizer = Optimizer(pipeline)
iterations = optimizer.tune_iter(dev_set, show_progress=False)
best_loss = 1.0
for i, iteration in enumerate(iterations):
    print(f"Best clustering threshold so far: {iteration['params']['clustering']['threshold']}")
    if i > iter_count:
        break


best_clustering_threshold = optimizer.best_params['clustering']['threshold']

finetuned_pipeline = SpeakerDiarization(
    segmentation=pretrained_model,
    embedding=pretrained_pipeline.embedding,
    embedding_exclude_overlap=pretrained_pipeline.embedding_exclude_overlap,
    clustering=pretrained_pipeline.klustering,
)

finetuned_pipeline.instantiate({
    "segmentation": {
        "threshold": best_segmentation_threshold,
        "min_duration_off": 0.0,
    },
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 15,
        "threshold": best_clustering_threshold,
    },
})

metric = DiarizationErrorRate()

for file in dataset.train():
    # apply finetuned pipeline
    file["finetuned pipeline"] = finetuned_pipeline(file)

    # evaluate its performance
    metric(file["annotation"], file["finetuned pipeline"], uem=file["annotated"])

new_hyperparameters = finetuned_pipeline.parameters(instantiated=True)

print('Base Params:', new_hyperparameters)
print(f"The finetuned pipeline reaches a Diarization Error Rate (DER) of {100 * abs(metric):.1f}% on {dataset.name} test set.")

finetuned_pipeline.dump_params(Path('hyperparameters/config.yaml'))