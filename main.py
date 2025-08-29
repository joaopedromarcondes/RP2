import whisper
from deepface import DeepFace
import cv2
import json
import os
import librosa
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

# ========================
# 1. Transcrição de Áudio (Whisper)
# ========================
def transcrever_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# ========================
# 2. Análise de Sentimento do Texto (VADER)
# ========================
def analisar_texto(texto):
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(texto)
    return scores

# ========================
# 3. Análise de Tom de Voz (Librosa)
# ========================
def analisar_voz(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)

    # Extração de features comuns
    energy = np.mean(librosa.feature.rms(y=y))
    zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    features = {
        "energy": float(energy),
        "zero_crossing_rate": float(zero_crossing_rate),
        "spectral_centroid": float(spectral_centroid)
    }
    return features


# ========================
# 4. Análise de Expressão Facial (DeepFace)
# ========================
def analisar_expressao(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            emotions.append(analysis[0]["dominant_emotion"])
        except:
            continue
    cap.release()

    if emotions:
        return max(set(emotions), key=emotions.count)
    return "indefinido"

# ========================
# 5. Integração e Exportação
# ========================
def analisar_entrevista(audio_path, video_path):
    resultado = {}

    # Transcrição
    texto = transcrever_audio(audio_path)
    resultado["transcricao"] = texto

    # Sentimento do texto
    resultado["sentimento"] = analisar_texto(texto)

    # Análise de voz
    resultado["voz"] = analisar_voz(audio_path)

    # Análise facial
    resultado["emocao_facial"] = analisar_expressao(video_path)

    # Salvar em JSON
    with open("resultado.json", "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=4)

    # Salvar em CSV
    df = pd.json_normalize(resultado)
    df.to_csv("resultado.csv", index=False)

    return resultado

# ========================
# EXECUTAR (exemplo)
# ========================
if __name__ == "__main__":
    audio = "saida.wav"  # coloque seu arquivo de áudio aqui
    video = "saida.mp4"  # coloque seu arquivo de vídeo aqui

    if os.path.exists(audio) and os.path.exists(video):
        res = analisar_entrevista(audio, video)
        print(json.dumps(res, indent=4, ensure_ascii=False))
    else:
        print("Coloque arquivos 'exemplo_audio.wav' e 'exemplo_video.mp4' na pasta!")