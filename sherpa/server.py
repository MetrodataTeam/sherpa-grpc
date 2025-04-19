import asyncio
from enum import StrEnum
from io import BytesIO
import logging
import os
from time import time
from typing import List

from audio import decode_audio
from data import OfflineRecognitionResult as _OfflineRecognitionResult
from data import (
  OfflineSpeakerDiarizationSegment as _OfflineSpeakerDiarizationSegment,
)
from grpclib.const import Status
from grpclib.exceptions import GRPCError
from grpclib.health.service import Health
from grpclib.reflection.service import ServerReflection
from grpclib.server import Server
from grpclib.utils import graceful_exit
import numpy as np
from pydantic import Field
from pydantic import model_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict
import sherpa_onnx

from pb.sherpa import Audio
from pb.sherpa import DiarizationConfig
from pb.sherpa import OfflineRecognitionResult
from pb.sherpa import OfflineSpeakerDiarizationResult
from pb.sherpa import OfflineSpeakerDiarizationSegment
from pb.sherpa import PunctuationConfig
from pb.sherpa import SherpaServiceBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
logger.addHandler(console_handler)
for handler in logger.handlers:
  handler.setFormatter(
    logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  )


class Provider(StrEnum):
  cuda = 'cuda'
  cpu = 'cpu'
  coreml = 'coreml'
  xnnpack = 'xnnpack'
  nnapi = 'nnapi'
  trt = 'trt'
  directml = 'directml'


class OfflineRecognizerVendor(StrEnum):
  sensevoice = 'sensevoice'
  whisper = 'whisper'
  firered = 'firered'


class VadVendor(StrEnum):
  silero = 'silero'


class SegmentationVendor(StrEnum):
  pyannote = 'pyannote'


class EmbeddingVendor(StrEnum):
  speaker_3d = '3dspeaker'


class PunchuationVendor(StrEnum):
  # https://huggingface.co/funasr/ct-punc
  ct_punc = 'ct_punc'


class Settings(BaseSettings):
  """TODO(Deo): add a pipeline config so we could done diarization/recognition/punctuation in one go"""

  model_config = SettingsConfigDict(env_prefix='sherpa_')
  execution_provider: Provider = Field(
    Provider.cuda, description='onnx execution provider'
  )
  num_threads: int = Field(1, description='onnx execution provider num threads')
  offline_recognizer_vendor: OfflineRecognizerVendor = Field(
    OfflineRecognizerVendor.sensevoice, description='offline recognizer vendor'
  )
  offline_recognizer_model: str = Field(
    '', description='huggingface hub compatible model id'
  )
  segmentation_vendor: SegmentationVendor = Field(
    SegmentationVendor.pyannote, description='segmentation vendor'
  )
  segmentation_model: str = Field('', description='segmentation model')
  embedding_vendor: EmbeddingVendor = Field(
    EmbeddingVendor.speaker_3d, description='embedding extractor vendor'
  )
  embedding_model: str = Field('', description='embedding extractor model')
  vad_vendor: VadVendor = Field(VadVendor.silero, description='vad vendor')
  vad_model: str = Field('', description='vad model')
  punctuation_vendor: PunchuationVendor = Field(
    PunchuationVendor.ct_punc, description='punctuation vendor'
  )
  punctuation_model: str = Field('', description='punctuation model')
  feature_offline_recognizer: bool = Field(False)
  feature_offline_speaker_diarization: bool = Field(False)
  feature_vad: bool = Field(False)
  feature_punctuation: bool = Field(False)
  host: str = Field('0.0.0.0', description='listen host')
  port: int = Field(18913, description='listen port')
  debug: bool = Field(False)

  @model_validator(mode='after')
  def check_feature(self):
    if self.feature_offline_recognizer:
      if not self.offline_recognizer_model:
        raise ValueError('offline recognizer model is required')
    if self.feature_offline_speaker_diarization:
      if not self.segmentation_model:
        raise ValueError('segmentation model is required')
      if not self.embedding_model:
        raise ValueError('embedding model is required')
    return self


class SherpaService(SherpaServiceBase):
  offline_recognizer_model = None
  diarization_model = None
  vad_model: sherpa_onnx.VoiceActivityDetector = None
  punctuation_model = None

  def __init__(self, settings: Settings):
    if settings.feature_offline_recognizer:
      model_directory = os.path.dirname(settings.offline_recognizer_model)
      match settings.offline_recognizer_vendor:
        case OfflineRecognizerVendor.sensevoice:
          tokens_path = os.path.join(model_directory, 'tokens.txt')
          self.offline_recognizer_model = (
            sherpa_onnx.OfflineRecognizer.from_sense_voice(
              settings.offline_recognizer_model,
              tokens_path,
              provider=settings.execution_provider,
              num_threads=settings.num_threads,
              language='zh',
              use_itn=True,
              debug=settings.debug,
            )
          )
        case OfflineRecognizerVendor.firered:
          encoder_path = os.path.join(model_directory, 'encoder.int8.onnx')
          decoder_path = os.path.join(model_directory, 'decoder.int8.onnx')
          tokens_path = os.path.join(model_directory, 'tokens.txt')
          self.offline_recognizer_model = (
            sherpa_onnx.OfflineRecognizer.from_fire_red_asr(
              encoder_path,
              decoder_path,
              tokens_path,
              provider=settings.execution_provider,
              num_threads=settings.num_threads,
              debug=settings.debug,
            )
          )
        case _:
          raise NotImplementedError(
            f'unsupported offline recognizer vendor {settings.offline_recognizer_vendor}'
          )
    if settings.feature_offline_speaker_diarization:
      match settings.segmentation_vendor:
        case SegmentationVendor.pyannote:
          segmentation_config = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
              model=settings.segmentation_model,
            ),
            provider=settings.execution_provider,
            num_threads=settings.num_threads,
            debug=settings.debug,
          )
        case _:
          raise NotImplementedError(
            f'unsupported segmentation vendor {settings.segmentation_vendor}'
          )
      self.diarization_model = sherpa_onnx.OfflineSpeakerDiarization(
        sherpa_onnx.OfflineSpeakerDiarizationConfig(
          segmentation=segmentation_config,
          embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=settings.embedding_model,
            provider=settings.execution_provider,
            num_threads=settings.num_threads,
            debug=settings.debug,
          ),
          clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=-1,
            threshold=0.5,
          ),
          min_duration_on=0.3,
          min_duration_off=0.5,
        )
      )
    if settings.feature_vad:
      match settings.vad_vendor:
        case VadVendor.silero:
          # https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/python/csrc/silero-vad-model-config.cc
          config = sherpa_onnx.VadModelConfig(
            silero_vad=sherpa_onnx.SileroVadModelConfig(
              model=settings.vad_model,
              min_silence_duration=0.25,
              min_speech_duration=0.25,
            ),
            # using silero v5 works on gpu, but much slower
            # provider=settings.execution_provider,
            num_threads=settings.num_threads,
            debug=settings.debug,
          )

          self.vad_model = sherpa_onnx.VoiceActivityDetector(
            config, buffer_size_in_seconds=100
          )
        case _:
          raise NotImplementedError(
            f'unsupported vad vendor {settings.vad_vendor}'
          )
    if settings.feature_punctuation:
      match settings.punctuation_vendor:
        case PunchuationVendor.ct_punc:
          config = sherpa_onnx.OfflinePunctuationConfig(
            model=sherpa_onnx.OfflinePunctuationModelConfig(
              ct_transformer=settings.punctuation_model,
              provider=settings.execution_provider,
              num_threads=settings.num_threads,
              debug=settings.debug,
            )
          )
          self.punctuation_model = sherpa_onnx.OfflinePunctuation(config)
        case _:
          raise NotImplementedError(
            f'unsupported punctuation vendor {settings.punctuation_vendor}'
          )

  async def vad(self, request: Audio) -> OfflineSpeakerDiarizationResult:
    if self.vad_model is None:
      raise GRPCError(Status.FAILED_PRECONDITION, 'vad model not loaded')
    start = time()
    if request.is_numpy_data:
      data = np.frombuffer(request.data, dtype=np.float32).copy()
    else:
      data = decode_audio(BytesIO(request.data))
    window_size = self.vad_model.config.silero_vad.window_size
    self.vad_model.reset()
    while len(data) > window_size:
      self.vad_model.accept_waveform(data[:window_size])
      data = data[window_size:]
    segments = []
    while not self.vad_model.empty():
      segment = self.vad_model.front
      segments.append(
        OfflineSpeakerDiarizationSegment(
          start=segment.start / 16000,
          end=(segment.start + len(segment.samples)) / 16000,
          speaker=0,
          text='',
        )
      )
      self.vad_model.pop()
    self.vad_model.flush()
    while not self.vad_model.empty():
      segment = self.vad_model.front
      segments.append(
        OfflineSpeakerDiarizationSegment(
          start=segment.start / 16000,
          end=(segment.start + len(segment.samples)) / 16000,
          speaker=0,
          text='',
        )
      )
      self.vad_model.pop()
    time_span = time() - start
    logger.info(
      'finish vad %s in %.3f ms', request.info or 'data', time_span * 1000
    )
    return OfflineSpeakerDiarizationResult(segments=segments)

  async def offline_recognize(self, request: Audio) -> OfflineRecognitionResult:
    if self.offline_recognizer_model is None:
      raise GRPCError(Status.FAILED_PRECONDITION, 'recornizer model not loaded')
    punctuation = request.punctuation or PunctuationConfig()
    start = time()
    if request.is_numpy_data:
      data = np.frombuffer(request.data, dtype=np.float32).copy()
    else:
      data = decode_audio(BytesIO(request.data))
    stream = self.offline_recognizer_model.create_stream()
    stream.accept_waveform(16000, data)
    self.offline_recognizer_model.decode_stream(stream)
    result: _OfflineRecognitionResult = stream.result
    language = result.lang[2:-2]
    text = result.text
    if punctuation.enable and self.punctuation_model and text:
      text = self.punctuation_model.add_punctuation(result.text)
    # TODO(Deo): implement segments
    # https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/generate-subtitles.py#L494C5-L581C19
    res = OfflineRecognitionResult(
      lang=language,
      emotion=result.emotion,
      event=result.event,
      text=text,
      timestamps=result.timestamps,
      tokens=result.tokens,
      words=result.words,
    )
    time_span = time() - start
    if result.timestamps:
      logger.info(
        'finish transcribe %s in %.3f ms, duration: %.2fs, ratio %.2f, language %s',
        request.info or 'data',
        time_span * 1000,
        result.timestamps[-1],
        result.timestamps[-1] / time_span,
        language,
      )
    else:
      # firered has no timestamps info at the moment
      logger.info(
        'finish transcribe %s in %.3f ms, language %s',
        request.info or 'data',
        time_span * 1000,
        language,
      )
    return res

  async def offline_speaker_diarization(
    self, request: Audio
  ) -> OfflineSpeakerDiarizationResult:
    """
    https://github.com/k2-fsa/sherpa-onnx/issues/1708 accuracy problem
    """
    if self.diarization_model is None:
      raise GRPCError(
        Status.FAILED_PRECONDITION, 'diaorization model not loaded'
      )
    diarization = request.diarization or DiarizationConfig()
    punctuation = request.punctuation or PunctuationConfig()
    if diarization.recognize and self.offline_recognizer_model is None:
      raise GRPCError(Status.FAILED_PRECONDITION, 'recornizer model not loaded')
    # update config
    # https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/c-api/c-api.h#L1548
    num_clusters = -1
    if diarization.clustering_num_clusters is not None:
      num_clusters = diarization.clustering_num_clusters
    threshold = 0.5
    if diarization.clustering_threshold is not None:
      threshold = diarization.clustering_threshold
    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
      segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(),
      embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(),
      clustering=sherpa_onnx.FastClusteringConfig(
        num_clusters=num_clusters, threshold=threshold
      ),
      min_duration_on=0.3,
      min_duration_off=0.5,
    )
    self.diarization_model.set_config(config)
    start = time()
    if request.is_numpy_data:
      data = np.frombuffer(request.data, dtype=np.float32).copy()
    else:
      data = decode_audio(BytesIO(request.data))
    result = self.diarization_model.process(data).sort_by_start_time()
    segments = []
    for i in result:
      i: _OfflineSpeakerDiarizationSegment
      text = ''
      if diarization.recognize and self.offline_recognizer_model is not None:
        stream = self.offline_recognizer_model.create_stream()
        stream.accept_waveform(
          16000, data[int(i.start * 16000) : int(i.end * 16000)]
        )
        self.offline_recognizer_model.decode_stream(stream)
        _result: _OfflineRecognitionResult = stream.result
        text = _result.text
        if punctuation.enable and self.punctuation_model is not None and text:
          text = self.punctuation_model.add_punctuation(text)
      segments.append(
        OfflineSpeakerDiarizationSegment(
          start=i.start,
          end=i.end,
          speaker=i.speaker,
          text=text,
        )
      )
    res = OfflineSpeakerDiarizationResult(segments=segments)
    time_span = time() - start
    if result:
      logger.info(
        'finish diarization %s in %.3f ms, duration: %.2fs, ratio %.2f',
        request.info or 'data',
        time_span * 1000,
        result[-1].end,
        result[-1].end / time_span,
      )
    return res


def get_sherpa_services(*args, **kwargs) -> List:
  return ServerReflection.extend([SherpaService(*args, **kwargs), Health()])


async def serve(settings: Settings):
  server = Server(get_sherpa_services(settings))
  with graceful_exit([server]):
    await server.start(settings.host, settings.port)
    logger.info(
      'listen on %s:%d, using %s execution provider',
      settings.host,
      settings.port,
      settings.execution_provider,
    )
    if settings.feature_offline_recognizer:
      logger.info(
        'recognition model: %s %s',
        settings.offline_recognizer_vendor,
        settings.offline_recognizer_model,
      )
    if settings.feature_offline_speaker_diarization:
      logger.info(
        'diarization model: %s %s %s %s',
        settings.segmentation_vendor,
        settings.segmentation_model,
        settings.embedding_vendor,
        settings.embedding_model,
      )
    if settings.feature_vad:
      logger.info('vad model: %s %s', settings.vad_vendor, settings.vad_model)
    if settings.feature_punctuation:
      logger.info(
        'punctuation model: %s %s',
        settings.punctuation_vendor,
        settings.punctuation_model,
      )
    await server.wait_closed()
    logger.info('Goodbye!')


if __name__ == '__main__':
  _settings = Settings()
  asyncio.run(serve(_settings))
