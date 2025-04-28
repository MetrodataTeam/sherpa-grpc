## generate pb
```bash
# use betterproto, don't use pydantic before it's compatible with v2
# move to multi stage docker build to avoid submit it to git
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=sherpa/pb/ pb/sherpa.proto
python -m grpc_tools.protoc -I ./pb --python_betterproto_out=pb/ pb/sherpa.proto
```

## local test
```bash
cd sherpa
# offline recognize
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_FEATURE_OFFLINE_RECOGNIZER=1 SHERPA_OFFLINE_RECOGNIZER_MODEL=~/Downloads/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx pytest -vv tests/server_test.py
# offline speaker diarazation
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_FEATURE_OFFLINE_SPEAKER_DIARIZATION=1 SHERPA_SEGMENTATION_MODEL=~/Downloads/sherpa-onnx-pyannote-segmentation-3-0/model.onnx SHERPA_EMBEDDING_MODEL=~/Downloads/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx pytest -vv tests/server_test.py
# offline speaker diarazation with recognizer
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_FEATURE_OFFLINE_RECOGNIZER=1 SHERPA_FEATURE_OFFLINE_SPEAKER_DIARIZATION=1 SHERPA_OFFLINE_RECOGNIZER_MODEL=~/Downloads/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx SHERPA_SEGMENTATION_MODEL=~/Downloads/sherpa-onnx-pyannote-segmentation-3-0/model.onnx SHERPA_EMBEDDING_MODEL=~/Downloads/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx pytest -vv tests/server_test.py

```

## local inference
```bash
cd sherpa
poetry install
# poetry.lock use a gpu version, we need to install it separately(usually cpu version)
pip install sherpa-onnx
# offline recognize, we use sensevoice model by default
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_OFFLINE_RECOGNIZER_MODEL=~/Downloads/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx python server.py
# offline speaker diarazation, using pyannote and 3dspeaker
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_OFFLINE_RECOGNIZER_MODEL=~/Downloads/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/model.onnx SHERPA_SEGMENTATION_MODEL=~/Downloads/sherpa-onnx-pyannote-segmentation-3-0/model.onnx SHERPA_EMBEDDING_MODEL=~/Downloads/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx SHERPA_FEATURE_offline_speaker_diarization=1 SHERPA_FEATURE_OFFLINE_RECOGNIZER=1 python server.py
# xiaohongshu with punctuation
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_OFFLINE_RECOGNIZER_MODEL=~/Downloads/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/ SHERPA_OFFLINE_RECOGNIZER_VENDOR=firered SHERPA_SEGMENTATION_MODEL=~/Downloads/sherpa-onnx-pyannote-segmentation-3-0/model.onnx SHERPA_EMBEDDING_MODEL=~/Downloads/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx SHERPA_FEATURE_offline_speaker_diarization=1 SHERPA_FEATURE_OFFLINE_RECOGNIZER=1 SHERPA_FEATURE_PUNCTUATION=1 SHERPA_PUNCTUATION_MODEL=~/Downloads/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx python server.py
# speaker identification
SHERPA_EXECUTION_PROVIDER=cpu SHERPA_FEATURE_SPEAKER_IDENTIFICATION=1 SHERPA_SPEAKER_IDENTIFICATION_MODEL=~/Downloads/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx  python server.py
```

# docker build
```bash
docker build --build-arg BASE_IMAGE=python:3.12.10-slim --build-arg PYPI="" -t sherpa-grpc:test -f Dockerfile.cpu .
```

## configuration
| environment               | default | comment     |
| ------------------------- | ------- | ----------- |
| SHERPA_EXECUTION_PROVIDER | cuda    | cuda/cpu    |
| SHERPA_HOST               | 0.0.0.0 | listen host |
| SHERPA_PORT               | 18913   | listen port |
