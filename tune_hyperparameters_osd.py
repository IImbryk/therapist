# from pyannote.audio.pipelines import OverlappedSpeechDetection
# from pyannote.audio import Model
#
# model = Model.from_pretrained("pyannote/segmentation", use_auth_token='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')
# overlapp_pipeline = OverlappedSpeechDetection(segmentation=model)
# overlapp_pipeline.instantiate({"onset": 0.5, "offset": 0.5, "min_duration_on": 0.1, "min_duration_off": 0.1})