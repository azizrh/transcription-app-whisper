# from flask import Flask, render_template, request, jsonify
# from models.translator import TranslationManager
# from models.whisper_model import WhisperModel
# from utils.audio import process_audio_file
# import os

# app = Flask(__name__)

# # Initialize models
# print("Loading models...")
# whisper_model = WhisperModel()
# translator = TranslationManager()
# print("Models loaded!")

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/process_audio', methods=['POST'])
# def process_audio():
#     try:
#         # Get the audio data and language
#         audio_data = request.json.get('audio')
#         target_lang = request.json.get('target_lang', 'es')
        
#         # Process audio file
#         wav_path, temp_audio_path = process_audio_file(audio_data)
        
#         # Transcribe
#         result = whisper_model.transcribe(wav_path)
#         transcription = result["text"].strip()

#         # Translate
#         translation = translator.translate(transcription, target_lang)

#         # Cleanup
#         os.unlink(temp_audio_path)
#         os.unlink(wav_path)

#         return jsonify({
#             'transcription': transcription,
#             'translation': translation
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, render_template, request, jsonify

# from translator import TranslationManager

# from whisper_model import WhisperModel

# from audio_utils import process_audio_file

from models.translator import TranslationManager
from models.whisper_model import WhisperModel
from utils.audio import process_audio_file

import os

import numpy as np
 
app = Flask(__name__)
 
# Initialize models with smaller model for faster inference

print("Loading models...")

whisper_model = WhisperModel()  # Changed to tiny for faster processing

translator = TranslationManager()

print("Models loaded!")
 
@app.route('/')

def index():

    return render_template('index.html')
 
@app.route('/process_audio', methods=['POST'])

def process_audio():

    try:

        audio_data = request.json.get('audio')

        target_lang = request.json.get('target_lang', 'es')

        # Process audio file

        wav_path, temp_audio_path = process_audio_file(audio_data)

        # Transcribe with shorter segments

        result = whisper_model.transcribe(wav_path)

        transcription = result["text"].strip()
 
        # Only translate if there's actual content

        if transcription:

            translation = translator.translate(transcription, target_lang)

        else:

            translation = ""
 
        # Cleanup

        os.unlink(temp_audio_path)

        os.unlink(wav_path)
 
        return jsonify({

            'transcription': transcription,

            'translation': translation

        })
 
    except Exception as e:

        print(f"Error: {str(e)}")

        return jsonify({'error': str(e)}), 500
 
if __name__ == '__main__':

    app.run(host='0.0.0.0', port=5000)
 