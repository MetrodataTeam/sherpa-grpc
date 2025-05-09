"""copy from faster-whisper"""

import gc
import io
import itertools
from typing import BinaryIO, Union

import av
import numpy as np
import soundfile as sf
from grpclib.const import Status
from grpclib.exceptions import GRPCError


def convert_input_bytes_to_wav_bytes(input_bytes) -> bytes:
  # 创建一个 BytesIO 对象用于读取输入字节流
  # 创建一个 BytesIO 对象用于写入 WAV 字节流
  with io.BytesIO(input_bytes) as input_buffer, io.BytesIO() as output_buffer:
    # 打开输入容器（自动检测格式）
    with (
      av.open(input_buffer) as input_container,
      av.open(output_buffer, mode='w', format='wav') as output_container,
    ):
      # 添加音频流，使用 PCM 编码
      output_stream = output_container.add_stream('pcm_s16le')

      # 解码输入音频帧并编码到输出容器
      for frame in input_container.decode(audio=0):
        frame.pts = None  # 重置时间戳
        for packet in output_stream.encode(frame):
          output_container.mux(packet)

      # 刷新编码器
      for packet in output_stream.encode(None):
        output_container.mux(packet)

    # 获取 WAV 字节流
    output_buffer.seek(0)
    wav_bytes = output_buffer.read()

  return wav_bytes


def load_audio(blob: bytes) -> tuple[np.ndarray, int]:
  """load audio file"""
  blob = convert_input_bytes_to_wav_bytes(blob)
  with io.BytesIO(blob) as f:
    try:
      data, sample_rate = sf.read(
        f,
        always_2d=True,
        dtype='float32',
      )
    except sf.LibsndfileError:
      raise GRPCError(
        Status.INVALID_ARGUMENT,
        message='Invalid audio file',
      )
  # use only the first channel
  data = data[:, 0]
  samples = np.ascontiguousarray(data)
  return samples, sample_rate


def decode_audio(
  input_file: Union[str, BinaryIO],
  sampling_rate: int = 16000,
  split_stereo: bool = False,
):
  """Decodes the audio.

  Args:
    input_file: Path to the input file or a file-like object.
    sampling_rate: Resample the audio to this sample rate.
    split_stereo: Return separate left and right channels.

  Returns:
    A float32 Numpy array.

    If `split_stereo` is enabled, the function returns a 2-tuple with the
    separated left and right channels.
  """
  resampler = av.audio.resampler.AudioResampler(
    format='s16',
    layout='mono' if not split_stereo else 'stereo',
    rate=sampling_rate,
  )

  raw_buffer = io.BytesIO()
  dtype = None

  with av.open(input_file, mode='r', metadata_errors='ignore') as container:
    frames = container.decode(audio=0)
    frames = _ignore_invalid_frames(frames)
    frames = _group_frames(frames, 500000)
    frames = _resample_frames(frames, resampler)

    for frame in frames:
      array = frame.to_ndarray()
      dtype = array.dtype
      raw_buffer.write(array)

  # It appears that some objects related to the resampler are not freed
  # unless the garbage collector is manually run.
  # https://github.com/SYSTRAN/faster-whisper/issues/390
  # note that this slows down loading the audio a little bit
  # if that is a concern, please use ffmpeg directly as in here:
  # https://github.com/openai/whisper/blob/25639fc/whisper/audio.py#L25-L62
  del resampler
  gc.collect()

  audio = np.frombuffer(raw_buffer.getbuffer(), dtype=dtype)

  # Convert s16 back to f32.
  audio = audio.astype(np.float32) / 32768.0

  if split_stereo:
    left_channel = audio[0::2]
    right_channel = audio[1::2]
    return left_channel, right_channel

  return audio


def _ignore_invalid_frames(frames):
  iterator = iter(frames)

  while True:
    try:
      yield next(iterator)
    except StopIteration:
      break
    except av.error.InvalidDataError:
      continue


def _group_frames(frames, num_samples=None):
  fifo = av.audio.fifo.AudioFifo()

  for frame in frames:
    frame.pts = None  # Ignore timestamp check.
    fifo.write(frame)

    if num_samples is not None and fifo.samples >= num_samples:
      yield fifo.read()

  if fifo.samples > 0:
    yield fifo.read()


def _resample_frames(frames, resampler):
  # Add None to flush the resampler.
  for frame in itertools.chain(frames, [None]):
    yield from resampler.resample(frame)
