from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
from models.translator import TranslationManager
from models.whisper_model import WhisperModel
from utils.audio import process_audio_file
 
import numpy as np
 
app = Flask(__name__)
 
# Initialize Flask-SocketIO
socketio = SocketIO(app, async_mode='eventlet')
 
# Initialize models
print("Loading models...")
whisper_model = WhisperModel()  # Changed to tiny for faster processing
translator = TranslationManager()
print("Models loaded!")
 
# HTTP route for the main page
@app.route('/')
def index():
    return render_template('index2.html')
 
# POST route to process audio
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
 
        # Translate if transcription exists
        translation = translator.translate(transcription, target_lang) if transcription else ""
 
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
 
# WebSocket route to handle audio streaming
@socketio.on('connect')
def handle_connect():
    print('Client connected via WebSocket')
    emit('message', 'Connected to WebSocket server!')
 
# WebSocket route for processing audio
@socketio.on('audio_data')
def handle_audio_data(data):
    try:
        # Expecting the audio data as a base64 encoded string or binary
        audio_data = data.get('audio')  # This should be the binary data (Base64 or Raw)
        target_lang = data.get('target_lang', 'es')
 
        # Process audio file
        wav_path, temp_audio_path = process_audio_file(audio_data)
 
        # Transcribe with shorter segments
        result = whisper_model.transcribe(wav_path)
        transcription = result["text"].strip()
 
        # Translate if transcription exists
        translation = translator.translate(transcription, target_lang) if transcription else ""
 
        # Cleanup
        os.unlink(temp_audio_path)
        os.unlink(wav_path)
 
        # Send transcription and translation back to the client
        emit('audio_processed', {
            'transcription': transcription,
            'translation': translation
        })
 
    except Exception as e:
        print(f"Error in WebSocket: {str(e)}")
        emit('error', {'error': str(e)})
 
# Handle WebSocket disconnection
@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected via WebSocket')
 
if __name__ == '__main__':
    # Run Flask with WebSocket support
    socketio.run(app, host='0.0.0.0', port=5000)