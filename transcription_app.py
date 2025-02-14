import numpy as np
import pyaudio
import whisper
import threading
import queue
import tkinter as tk
from tkinter import ttk, scrolledtext
import time
from datetime import datetime
 
class TranscriptionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-time Transcription")
        self.root.geometry("800x600")
        
        # Initialize audio parameters
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.buffer_duration = 2.0  # seconds
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # Initialize Whisper model
        print("Loading Whisper model...")
        self.model = whisper.load_model("turbo", device="cuda")
        print("Model loaded!")
        
        # Initialize audio processing components
        self.audio = pyaudio.PyAudio()
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        self.text_queue = queue.Queue()
        self.is_recording = False
        
        self.setup_gui()
        self.update_transcription()
 
    def setup_gui(self):
        # Create main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Control frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        control_frame.columnconfigure(1, weight=1)
        
        # Buffer duration control
        ttk.Label(control_frame, text="Buffer Duration (s):").grid(row=0, column=0, padx=(0, 5))
        self.buffer_var = tk.StringVar(value="2.0")
        buffer_entry = ttk.Entry(control_frame, textvariable=self.buffer_var, width=10)
        buffer_entry.grid(row=0, column=1, sticky=(tk.W))
        
        # Start/Stop button
        self.record_button = ttk.Button(control_frame, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=0, column=2, padx=5)
        
        # Clear button
        self.clear_button = ttk.Button(control_frame, text="Clear", command=self.clear_text)
        self.clear_button.grid(row=0, column=3)
        
        # Status indicator
        self.status_var = tk.StringVar(value="Status: Ready")
        status_label = ttk.Label(control_frame, textvariable=self.status_var)
        status_label.grid(row=0, column=4, padx=5)
        
        # Transcription display
        self.text_display = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=30)
        self.text_display.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.rowconfigure(1, weight=1)
        
        # Apply some styling
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TFrame', padding=5)
 
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
        
        # Start audio stream
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
 
    def clear_text(self):
        self.text_display.delete(1.0, tk.END)
 
    def audio_callback(self, in_data, frame_count, time_info, status):
        if self.is_recording:
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Calculate remaining space in buffer
            space_left = self.buffer_size - self.buffer_index
            
            if space_left >= len(audio_data):
                # If enough space, just add the data
                self.audio_buffer[self.buffer_index:self.buffer_index + len(audio_data)] = audio_data
                self.buffer_index += len(audio_data)
            else:
                # If not enough space, fill what we can and process the buffer
                self.audio_buffer[self.buffer_index:] = audio_data[:space_left]
                self.process_buffer()
                
                # Reset buffer and add remaining audio data
                self.buffer_index = len(audio_data) - space_left
                self.audio_buffer[:self.buffer_index] = audio_data[space_left:]
        
        return (in_data, pyaudio.paContinue)
 
    def process_buffer(self):
        # Create a copy of the buffer for processing
        audio_data = self.audio_buffer.copy()
        
        # Reset buffer
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.buffer_index = 0
        
        # Run inference in a separate thread
        threading.Thread(target=self._run_inference, args=(audio_data,)).start()
 
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
                self.text_queue.put(f"[{timestamp}] {result['text'].strip()}")
        except Exception as e:
            print(f"Error in inference: {e}")
 
    def update_transcription(self):
        try:
            while True:
                text = self.text_queue.get_nowait()
                self.text_display.insert(tk.END, text + "\n")
                self.text_display.see(tk.END)
        except queue.Empty:
            pass
        
        self.root.after(100, self.update_transcription)
 
if __name__ == "__main__":
    # Create main window
    root = tk.Tk()
    app = TranscriptionApp(root)
    
    # Start the GUI event loop
    root.mainloop()