import whisper
import torch

class WhisperModel:
    def __init__(self):
        self.model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path)