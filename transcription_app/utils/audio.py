from pydub import AudioSegment
import tempfile
import base64
import os

def process_audio_file(audio_data):
    """Process audio data from base64 to wav file"""
    try:
        # Decode base64 audio
        audio_binary = base64.b64decode(audio_data.split(',')[1])
        
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_audio:
            temp_audio.write(audio_binary)
            temp_audio_path = temp_audio.name

        # Convert to WAV
        audio = AudioSegment.from_file(temp_audio_path)
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            audio.export(temp_wav.name, format='wav')
            wav_path = temp_wav.name

        return wav_path, temp_audio_path

    except Exception as e:
        raise Exception(f"Error processing audio: {str(e)}")