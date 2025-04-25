from pydantic import BaseModel


class OfflineRecognitionResult(BaseModel):
  lang: str
  emotion: str
  event: str
  text: str
  timestamps: list[float]
  tokens: list[str]
  words: list[str]


class OfflineSpeakerDiarizationSegment(BaseModel):
  start: float
  end: float
  speaker: int
