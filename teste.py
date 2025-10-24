import whisper
import cv2
import json
import sys
import os
from deepface import DeepFace

print(whisper.available_models())

def analyze_first_frame(video_path, detector_backend="opencv"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir vídeo: {video_path}")
        return
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("Não foi possível ler o primeiro frame.")
        return

    # Converte BGR -> RGB (DeepFace espera RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Tenta analisar (se falhar, tenta sem enforce_detection)
    try:
        result = DeepFace.analyze(frame_rgb, actions=["emotion"], detector_backend=detector_backend, enforce_detection=True)
    except Exception as e:
        print("Primeira tentativa falhou:", e)
        try:
            result = DeepFace.analyze(frame_rgb, actions=["emotion"], detector_backend=detector_backend, enforce_detection=False)
        except Exception as e2:
            print("Análise falhou:", e2)
            return

    print(result)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python teste.py caminho_para_video")
    else:
        analyze_first_frame(sys.argv[1])
