import whisper
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from deepface import DeepFace
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import threading
import queue
import time

# -----------------------------
# Configurações iniciais
# -----------------------------
model = whisper.load_model("tiny")  # modelo leve
sia = SentimentIntensityAnalyzer()
cap = cv2.VideoCapture(0)
samplerate = 16000  # Whisper funciona bem com 16kHz
duration = 2  # segundos de cada bloco de áudio
frame_skip = 5  # analisa 1 a cada 5 frames
audio_queue = queue.Queue()

# -----------------------------
# Função de thread para transcrição
# -----------------------------
def transcrever_thread():
    while True:
        audio_file = audio_queue.get()
        if audio_file is None:
            break
        result = model.transcribe(audio_file)
        texto = result["text"]
        print("Transcrição:", texto)
        print("Sentimento:", sia.polarity_scores(texto))
        audio_queue.task_done()

# Inicia a thread
threading.Thread(target=transcrever_thread, daemon=True).start()

# -----------------------------
# Loop principal (vídeo + áudio)
# -----------------------------
frame_count = 0
try:
    while True:
        # --- Captura de áudio ---
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        # salva temporariamente para o Whisper
        sf.write("temp.wav", audio_data, samplerate)
        audio_queue.put("temp.wav")

        # --- Captura de vídeo ---
        ret, frame = cap.read()
        if not ret:
            continue

        frame_count += 1
        if frame_count % frame_skip == 0:
            try:
                analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
                print("Emoção facial:", analysis[0]["dominant_emotion"])
            except:
                print("Emoção facial: indefinido")

        time.sleep(0.1)  # pequeno delay para reduzir carga da CPU

except KeyboardInterrupt:
    print("Encerrando...")
finally:
    audio_queue.put(None)  # encerra thread de áudio
    cap.release()
