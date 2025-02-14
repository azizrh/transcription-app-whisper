import numpy as np
import pyaudio
import whisper
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog
from datetime import datetime
from transformers import MarianMTModel, MarianTokenizer
import torch
from pydub import AudioSegment
import os

# Rest of the code remains the same...

class TranslationManager:
    def __init__(self):
        self.language_models = {
            'es': 'Helsinki-NLP/opus-mt-en-es',
            'fr': 'Helsinki-NLP/opus-mt-en-fr',
            'de': 'Helsinki-NLP/opus-mt-en-de',
            'it': 'Helsinki-NLP/opus-mt-en-it',
            'id':'Helsinki-NLP/opus-mt-en-id'
        }
        self.current_model = None
        self.current_tokenizer = None
        self.target_lang = None

    def load_model(self, lang_code):
        if lang_code not in self.language_models:
            raise ValueError(f"Unsupported language: {lang_code}")
        
        if self.target_lang != lang_code:
            model_name = self.language_models[lang_code]
            self.current_tokenizer = MarianTokenizer.from_pretrained(model_name)
            self.current_model = MarianMTModel.from_pretrained(model_name)
            if torch.cuda.is_available():
                self.current_model = self.current_model.to('cuda')
            self.target_lang = lang_code

    def translate(self, text):
        if not self.current_model or not text.strip():
            return ""
        
        inputs = self.current_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            translated = self.current_model.generate(**inputs)
        return self.current_tokenizer.batch_decode(translated, skip_special_tokens=True)[0]

class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Transcription and Translation")
        self.root.geometry("800x600")  # Larger initial window size
        
        # Make root window responsive
        self.root.columnconfigure(0, weight=20)
        self.root.rowconfigure(0, weight=20)

        
        # Initialize components
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.buffer_duration = 2.0
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        print("Loading models...")
        self.model = whisper.load_model("turbo", device="cuda")
        self.translator = TranslationManager()
        print("Models loaded!")
        
        self.audio = pyaudio.PyAudio()
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.text_queue = queue.Queue()
        self.is_recording = False
        self.is_processing_file = False
        self.video_path = None
        
        self.setup_gui()
        self.update_transcription()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # File selection button
        self.file_button = ttk.Button(control_frame, text="Select Video File", 
                                    command=self.select_video_file)
        self.file_button.grid(row=0, column=0, padx=5)
        
        # File path display
        self.file_path_var = tk.StringVar(value="No file selected")
        ttk.Label(control_frame, textvariable=self.file_path_var).grid(row=0, column=1, padx=5)
        
        # Process file button
        self.process_button = ttk.Button(control_frame, text="Process Video", 
                                       command=self.process_video_file)
        self.process_button.grid(row=0, column=2, padx=5)
        
        # Add language selection
        ttk.Label(control_frame, text="Target Language:").grid(row=0, column=3, padx=5)
        self.lang_var = tk.StringVar(value="es")
        lang_combo = ttk.Combobox(control_frame, textvariable=self.lang_var, 
                                 values=['es', 'fr', 'de', 'it','id'])
        lang_combo.grid(row=0, column=4)
        
        # Microphone controls in second row
        ttk.Label(control_frame, text="Buffer Duration (s):").grid(row=1, column=0, padx=5, pady=5)
        self.buffer_var = tk.StringVar(value="2.0")
        buffer_entry = ttk.Entry(control_frame, textvariable=self.buffer_var, width=10)
        buffer_entry.grid(row=1, column=1)
        
        self.record_button = ttk.Button(control_frame, text="Start Recording", 
                                      command=self.toggle_recording)
        self.record_button.grid(row=1, column=2, padx=5)
        
        self.clear_button = ttk.Button(control_frame, text="Clear", command=self.clear_text)
        self.clear_button.grid(row=1, column=3)
        
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=1, column=4, padx=5)

# Text displays
        # Create a frame for text displays
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_frame.columnconfigure(0, weight=1)
        text_frame.columnconfigure(1, weight=1)
        text_frame.rowconfigure(0, weight=1)  # Make row expandable

        # Original text panel
        original_frame = ttk.LabelFrame(text_frame, text="Original Text")
        original_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        original_frame.rowconfigure(0, weight=1)  # Make frame expandable
        original_frame.columnconfigure(0, weight=1)
        
        self.text_display = scrolledtext.ScrolledText(
            original_frame, 
            wrap=tk.WORD, 
            font=('Arial', 11),
            width=40,  # Set a reasonable width
            height=25  # Increase height
        )
        self.text_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Translated text panel
        translated_frame = ttk.LabelFrame(text_frame, text="Translated Text")
        translated_frame.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        translated_frame.rowconfigure(0, weight=1)  # Make frame expandable
        translated_frame.columnconfigure(0, weight=1)
        
        self.translation_display = scrolledtext.ScrolledText(
            translated_frame, 
            wrap=tk.WORD, 
            font=('Arial', 11),
            width=40,  # Set a reasonable width
            height=25  # Increase height
        )
        self.translation_display.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure text display colors and styles
        text_style = {
            'background': '#FFFFFF',
            'foreground': '#000000',
            'insertbackground': '#000000'  # Cursor color
        }
        
        self.text_display.configure(**text_style)
        self.translation_display.configure(**text_style)

    def _run_inference(self, audio_data):
        try:
            result = self.model.transcribe(
                audio_data,
                language="en",
                fp16=False,
                no_speech_threshold=0.6
            )
            
            if result["text"].strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                original_text = f"[{timestamp}] {result['text'].strip()}"
                
                # Translate the text
                try:
                    self.translator.load_model(self.lang_var.get())
                    translated_text = self.translator.translate(result["text"].strip())
                    self.text_queue.put((original_text, translated_text))
                except Exception as e:
                    print(f"Translation error: {e}")
                    self.text_queue.put((original_text, "Translation error"))
        except Exception as e:
            print(f"Error in inference: {e}")

    def update_transcription(self):
        try:
            while True:
                original_text, translated_text = self.text_queue.get_nowait()
                self.text_display.insert(tk.END, original_text + "\n")
                self.text_display.see(tk.END)
                self.translation_display.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {translated_text}\n")
                self.translation_display.see(tk.END)
        except queue.Empty:
            pass
        
        self.root.after(100, self.update_transcription)

    def clear_text(self):
        self.text_display.delete(1.0, tk.END)
        self.translation_display.delete(1.0, tk.END)
            
    def toggle_recording(self):
        if not self.is_recording:
            try:
                new_duration = float(self.buffer_var.get())
                if new_duration <= 0:
                    raise ValueError("Buffer duration must be positive")
                self.buffer_duration = new_duration
                self.buffer_size = int(self.sample_rate * self.buffer_duration)
                self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
                self.start_recording()
            except ValueError as e:
                self.status_var.set(f"Error: {str(e)}")
                return
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_button.configure(text="Stop Recording")
        self.status_var.set("Status: Recording")
        
        self.stream = self.audio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self.audio_callback
        )

    def stop_recording(self):
        self.is_recording = False
        self.record_button.configure(text="Start Recording")
        self.status_var.set("Status: Stopped")
        
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()

    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            space_left = self.buffer_size - self.buffer_index
            
            if space_left >= len(audio_data):
                self.audio_buffer[self.buffer_index:self.buffer_index + len(audio_data)] = audio_data
                self.buffer_index += len(audio_data)
            else:
                self.audio_buffer[self.buffer_index:] = audio_data[:space_left]
                self.process_buffer()
                self.buffer_index = len(audio_data) - space_left
                self.audio_buffer[:self.buffer_index] = audio_data[space_left:]
        
        return (in_data, pyaudio.paContinue)

    def process_buffer(self):
        audio_data = self.audio_buffer.copy()
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        threading.Thread(target=self._run_inference, args=(audio_data,)).start()
    def select_video_file(self):
        self.video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv")]
        )
        if self.video_path:
            self.file_path_var.set(os.path.basename(self.video_path))
            self.status_var.set("Status: File selected")

    def process_video_file(self):
        if not self.video_path:
            self.status_var.set("Error: No video file selected")
            return
        
        if self.is_processing_file:
            self.status_var.set("Error: Already processing a file")
            return
        
        self.is_processing_file = True
        self.status_var.set("Status: Processing video file...")
        threading.Thread(target=self._process_video).start()

    def _process_video(self):
        try:
            self.status_var.set("Status: Extracting audio from video...")
            # Extract audio from video
            temp_audio_path = "temp_audio.wav"
            
            # Convert video to audio using pydub
            audio = AudioSegment.from_file(self.video_path)
            audio = audio.set_frame_rate(self.sample_rate)
            audio = audio.set_channels(1)  # Convert to mono
            audio.export(temp_audio_path, format="wav")
            
            self.status_var.set("Status: Processing audio...")
            # Read audio data using numpy directly
            audio_data = AudioSegment.from_wav(temp_audio_path)
            samples = np.array(audio_data.get_array_of_samples())
            
            # Convert to float32 and normalize
            audio_float = samples.astype(np.float32)
            audio_float = audio_float / (np.iinfo(samples.dtype).max)
            
            # Process in smaller chunks for better stability
            chunk_duration = 20  # seconds
            chunk_samples = int(self.sample_rate * chunk_duration)
            
            # Calculate total chunks
            total_chunks = (len(audio_float) + chunk_samples - 1) // chunk_samples
            
            for i in range(0, len(audio_float), chunk_samples):
                # Ensure we don't go out of bounds
                end_idx = min(i + chunk_samples, len(audio_float))
                chunk = audio_float[i:end_idx]
                
                # Pad the last chunk if necessary
                if len(chunk) < chunk_samples:
                    chunk = np.pad(chunk, (0, chunk_samples - len(chunk)), 'constant')
                
                self._run_inference(chunk)
                
                # Update progress
                current_chunk = i // chunk_samples + 1
                progress = (current_chunk / total_chunks) * 100
                self.status_var.set(f"Status: Processing... {progress:.1f}%")
                
            # Cleanup
            os.remove(temp_audio_path)
            
            self.status_var.set("Status: Video processing completed")
        except Exception as e:
            self.status_var.set(f"Error processing video: {str(e)}")
            print(f"Error in video processing: {e}")
            import traceback
            print(traceback.format_exc())  # This will print the full error traceback
        finally:
            self.is_processing_file = False
  # Keep other methods (toggle_recording, start_recording, stop_recording, audio_callback, process_buffer) the same


if __name__ == "__main__":
    root = tk.Tk()
    app = TranscriptionApp(root)
    root.mainloop()