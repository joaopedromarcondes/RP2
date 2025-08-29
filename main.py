import whisper
import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
from deepface import DeepFace
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# ========================
# 1. Configurações iniciais
# ========================
model = whisper.load_model("base")
sia = SentimentIntensityAnalyzer()
cap = cv2.VideoCapture(0)  # webcam
duration = 5  # segundos para cada trecho de áudio
samplerate = 44100

# ========================
# Loop principal em tempo real
# ========================
try:
    while True:
        # ---- Captura de áudio ----
        print("Gravando áudio...")
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
        sd.wait()
        sf.write("temp.wav", audio_data, samplerate)

        # ---- Transcrição ----
        result = model.transcribe("temp.wav")
        texto = result["text"]
        print("Transcrição:", texto)

        # ---- Análise de sentimento ----
        sentimento = sia.polarity_scores(texto)
        print("Sentimento:", sentimento)

        # ---- Captura de vídeo e análise facial ----
        ret, frame = cap.read()
        if not ret:
            continue
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emocao = analysis[0]["dominant_emotion"]
        except:
            emocao = "indefinido"
        print("Emoção facial:", emocao)

        # ---- Mostra o frame na tela ----
        cv2.putText(frame, f"{emocao}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
