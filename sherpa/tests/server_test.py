from functools import partial
import os
import wave

from grpclib.health.v1.health_grpc import HealthStub
from grpclib.health.v1.health_pb2 import HealthCheckRequest
from grpclib.health.v1.health_pb2 import HealthCheckResponse
from grpclib.reflection.v1.reflection_grpc import ServerReflectionStub
from grpclib.reflection.v1.reflection_pb2 import ServerReflectionRequest
from grpclib.testing import ChannelFor
import numpy as np
import pytest
from server import get_sherpa_services
from server import Settings

from pb.sherpa import Audio
from pb.sherpa import DiarizationConfig
from pb.sherpa import SherpaServiceStub

approx = partial(pytest.approx, abs=1e-5)
offline_recognize_test = bool(os.getenv('SHERPA_FEATURE_OFFLINE_RECOGNIZER'))
offline_speaker_diarization_test = bool(
  os.getenv('SHERPA_FEATURE_OFFLINE_SPEAKER_DIARIZATION')
)


@pytest.fixture(scope='module', autouse=True)
def anyio_backend():
  return 'asyncio'


@pytest.mark.skipif(not offline_recognize_test, reason='Run test only in local')
async def test_offline_recognize_service():
  settings = Settings()
  services = get_sherpa_services(settings)
  with open('tests/test.ogg', 'rb') as f:
    data = f.read()
  async with ChannelFor(services) as channel:
    stub = SherpaServiceStub(channel)
    health = HealthStub(channel)
    reflection = ServerReflectionStub(channel)

    response = await stub.offline_recognize(Audio(info='test.ogg', data=data))
    assert response.lang == 'zh'
    assert response.text == '音测试。'
    assert len(response.words) == 0

    # health
    response = await health.Check(HealthCheckRequest())
    assert response.status == HealthCheckResponse.SERVING

    # reflection
    response = await reflection.ServerReflectionInfo(
      [ServerReflectionRequest(file_containing_symbol='SHERPA')]
    )
    assert len(response) == 1
    # TODO(Deo): it's not found at the moment
    #   https://github.com/danielgtaylor/python-betterproto/issues/443
    # assert response[0].name == ''
    # assert response[0].package == ''

    # TODO(Deo): make it work
    # unary_unary as stream control
    # async with stub.transcribe.open() as stream:
    #   for _ in range(2):
    #     await stream.send_message(Audio(info='test.ogg', data=data))
    #     response = await stream.recv_message()
    #     assert response.info.language == 'zh'
    #     assert response.info.probability > 0.9
    #     assert response.segments == [
    #         Segment(start=0.0,
    #                 end=4.0,
    #                 text='银车是',
    #                 no_speech_prob=approx(0.14626722037792206))
    #     ]
    #     assert response.text == '银车是'


@pytest.mark.skipif(not offline_recognize_test, reason='Run test only in local')
async def test_offline_recognize_from_numpy():
  settings = Settings()
  services = get_sherpa_services(settings)
  with open('tests/test.wav', 'rb') as f:
    file_data = f.read()
  with wave.open('tests/test.wav', 'rb') as wf:
    audio_data = wf.readframes(wf.getnframes())
    if wf.getsampwidth() == 2:
      audio_array = np.frombuffer(audio_data, dtype=np.int16)
    elif wf.getsampwidth() == 4:
      audio_array = np.frombuffer(audio_data, dtype=np.int32)
    else:
      raise ValueError(f'Unsupported sample width: {wf.getsampwidth()} bytes')
    # https://stackoverflow.com/questions/76448210/how-to-feed-a-numpy-array-as-audio-for-whisper-model
    data = (audio_array.astype(np.float32) / 32768.0).tobytes()
  async with ChannelFor(services) as channel:
    stub = SherpaServiceStub(channel)

    # file data
    for audio in (
      Audio(info='test.wav', data=file_data),
      Audio(info='test.wav', data=data, is_numpy_data=True),
    ):
      response = await stub.offline_recognize(audio)
      assert response.lang == 'zh'
      assert response.tokens == ['锄', '禾', '日', '当', '午', '。']
      assert response.timestamps == [
        approx(2.5199999809265137),
        approx(2.759999990463257),
        approx(3.359999895095825),
        approx(3.7200000286102295),
        approx(3.8999998569488525),
        approx(8.34000015258789),
      ]
      assert response.text == '锄禾日当午。'


@pytest.mark.skipif(
  not offline_speaker_diarization_test, reason='Run test only in local'
)
async def test_offline_diarization():
  settings = Settings()
  services = get_sherpa_services(settings)
  with open('tests/test.ogg', 'rb') as f:
    data = f.read()
  async with ChannelFor(services) as channel:
    stub = SherpaServiceStub(channel)

    response = await stub.offline_speaker_diarization(
      Audio(info='test.ogg', data=data)
    )
    assert response.to_dict(include_default_values=True) == {
      'segments': [
        {
          'end': approx(4.215969085693359),
          'start': approx(1.988468885421753),
          'speaker': 0,
          'text': '',
        },
        {
          'end': approx(2.7140939235687256),
          'speaker': 1,
          'start': approx(2.309093952178955),
          'text': '',
        },
        {
          'end': approx(7.067843914031982),
          'speaker': 1,
          'start': approx(5.751594066619873),
          'text': '',
        },
        {
          'end': approx(10.577844619750977),
          'speaker': 1,
          'start': approx(8.738469123840332),
          'text': '',
        },
      ]
    }


@pytest.mark.skipif(
  not offline_speaker_diarization_test or not offline_recognize_test,
  reason='Run test only in local',
)
async def test_offline_diarization_with_recognizer():
  settings = Settings()
  services = get_sherpa_services(settings)
  with open('tests/test.ogg', 'rb') as f:
    data = f.read()
  async with ChannelFor(services) as channel:
    stub = SherpaServiceStub(channel)

    response = await stub.offline_speaker_diarization(
      Audio(
        info='test.ogg',
        data=data,
        diarization=DiarizationConfig(recognize=True),
      )
    )
    assert response.to_dict(include_default_values=True) == {
      'segments': [
        {
          'end': approx(4.215969085693359),
          'start': approx(1.988468885421753),
          'speaker': 0,
          'text': '您测试。',
        },
        {
          'end': approx(2.7140939235687256),
          'speaker': 1,
          'start': approx(2.309093952178955),
          'text': '嗯。',
        },
        {
          'end': approx(7.067843914031982),
          'speaker': 1,
          'start': approx(5.751594066619873),
          'text': '你你讲我的。',
        },
        {
          'end': approx(10.577844619750977),
          'speaker': 1,
          'start': approx(8.738469123840332),
          'text': '这个全准备上带到间。',
        },
      ]
    }
