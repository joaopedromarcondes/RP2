import whisper
import cv2
from deepface import DeepFace
from transformers import pipeline
import json

# ============================
# 1. Transcrever áudio com timestamps (Whisper)
# ============================
def transcrever_com_tempo(audio_path):
    model = whisper.load_model("turbo")
    result = model.transcribe(audio_path, word_timestamps=True)
    frases = []
    for seg in result["segments"]:
        frases.append({
            "texto": seg["text"].strip(),
            "inicio": seg["start"],
            "fim": seg["end"]
        })
    return frases

# ============================
# 2. Analisar emoções faciais por frame do vídeo
# ============================
def analisar_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        tempo = frame_num / fps  # tempo em segundos
        try:
            analysis = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )
            emotions.append({
                "tempo": tempo,
                "emocao": analysis[0]["dominant_emotion"]
            })
        except:
            pass
        frame_num += 1
    cap.release()
    return emotions

# ============================
# 3. Análise de sentimento em português (Transformers)
# ============================
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def analisar_sentimento(texto):
    try:
        result = sentiment_pipeline(texto[:512])  # cortar textos muito longos
        return result[0]
    except Exception as e:
        return {"label": "erro", "score": 0.0}

# ============================
# 4. Combinar frases + emoções faciais + sentimento
# ============================
def combinar(frases, emotions):
    resultados = []
    for frase in frases:
        # pegar emoções do vídeo dentro do intervalo da frase
        emocoes_frase = [
            e["emocao"] for e in emotions
            if frase["inicio"] <= e["tempo"] <= frase["fim"]
        ]
        if emocoes_frase:
            emocao_dominante = max(set(emocoes_frase), key=emocoes_frase.count)
        else:
            emocao_dominante = "indefinido"

        # análise de sentimento do texto
        sentimento = analisar_sentimento(frase["texto"])

        resultados.append({
            "texto": frase["texto"],
            "inicio": frase["inicio"],
            "fim": frase["fim"],
            "emocao_facial": emocao_dominante,
            "sentimento_texto": sentimento["label"],
            "confianca_sentimento": float(sentimento["score"])
        })
    return resultados

# ============================
# Execução
# ============================
if __name__ == "__main__":
    audio = "saida.wav"
    video = "saida.mp4"

    print(">> Transcrevendo áudio...")
    frases = transcrever_com_tempo(audio)

    print(">> Analisando vídeo...")
    emotions = analisar_video(video)

    print(">> Combinando resultados...")
    combinados = combinar(frases, emotions)

    for c in combinados:
        print(c)

    with open("resultados_sentimento.json", "w", encoding="utf-8") as f:
        json.dump(combinados, f, ensure_ascii=False, indent=4)

    print("Resultados salvos em resultados_sentimento.json")
