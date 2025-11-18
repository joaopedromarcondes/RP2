import os
import json
import subprocess
from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import cv2
from deepface import DeepFace
import whisper

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
)
from reportlab.lib.styles import getSampleStyleSheet

# ============================
# Configurações iniciais
# ============================
whisper_model = whisper.load_model("turbo")

MAP_EMOCOES = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "sad": 3,
    "neutral": 4,
    "happy": 5,
    "surprise": 6,
    "indefinido": -1
}

# ============================
# Utilitários
# ============================
def converter_para_wav(video_path: Path, wav_path: Path):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True)

def transcrever_com_tempo(audio_path):
    result = whisper_model.transcribe(audio_path, word_timestamps=True)
    return [
        {"texto": seg["text"].strip(), "inicio": seg["start"], "fim": seg["end"]}
        for seg in result["segments"]
    ]

def analisar_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    emotions = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tempo = frame_num / fps
        try:
            analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
            dominant = analysis[0]["dominant_emotion"] if isinstance(analysis, list) else analysis["dominant_emotion"]
            emotions.append({"tempo": tempo, "emocao": dominant})
        except:
            pass

        frame_num += 1

    cap.release()
    return emotions

def combinar(frases, emotions):
    resultados = []
    for frase in frases:
        emocoes_frase = [e["emocao"] for e in emotions if frase["inicio"] <= e["tempo"] <= frase["fim"]]
        emocao_dominante = max(set(emocoes_frase), key=emocoes_frase.count) if emocoes_frase else "indefinido"
        resultados.append({
            "texto": frase["texto"],
            "inicio": frase["inicio"],
            "fim": frase["fim"],
            "emocao_facial": emocao_dominante
        })
    return resultados

# ============================
# Gráfico
# ============================
# Criar lista ordenada só com emoções válidas >= 0
EMOCOES_ORDENADAS = [e for e, idx in sorted(MAP_EMOCOES.items(), key=lambda x: x[1]) if MAP_EMOCOES[e] >= 0]
Y_TICKS = [MAP_EMOCOES[e] for e in EMOCOES_ORDENADAS]

def salvar_grafico(emotions, out_path):
    tempos = [e["tempo"] for e in emotions]
    valores = [MAP_EMOCOES.get(e["emocao"], -1) for e in emotions]

    plt.figure(figsize=(12, 4))
    plt.plot(tempos, valores, linewidth=0.8)

    # ✅ Ajustar o eixo Y para mostrar nomes
    plt.yticks(Y_TICKS, EMOCOES_ORDENADAS)

    plt.xlabel("Tempo (s)")
    plt.ylabel("Emoção")
    plt.title("Evolução das emoções no vídeo")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

# ============================
# PDF
# ============================
def gerar_pdf_report(outdir: Path, base: str, freq_path: Path, combinado_path: Path, grafico_path: Path):
    pdf_path = outdir / f"{base}_report.pdf"
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, topMargin=2*cm)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph(f"Relatório — {base}", styles["Title"]))
    flow.append(Spacer(1, 0.4*cm))

    if grafico_path.exists():
        img = RLImage(str(grafico_path))
        max_width = A4[0] - 4*cm
        aspect = img.imageHeight / img.imageWidth
        img.drawWidth = max_width
        img.drawHeight = max_width * aspect
        flow.append(img)
        flow.append(Spacer(1, 0.4*cm))

    flow.append(Paragraph("Frequência de emoções", styles["Heading2"]))

    if freq_path.exists():
        with open(freq_path, "r", encoding="utf-8") as f:
            freq = json.load(f)
    else:
        freq = {}

    table_data = [["Emoção", "Contagem"]] + [[k, str(v)] for k, v in sorted(freq.items(), key=lambda x: -x[1])]
    tbl = Table(table_data)
    tbl.setStyle(TableStyle([("GRID", (0,0), (-1,-1), 0.5, colors.grey)]))
    flow.append(tbl)

    doc.build(flow)
    return pdf_path

# ============================
# Lote
# ============================

def processar_video_unico(nome_arquivo):
    pasta = Path("entrevistas")
    if not pasta.exists():
        raise FileNotFoundError("Pasta 'entrevistas' não existe.")

    video = pasta / nome_arquivo

    if not video.exists():
        raise FileNotFoundError(f"Arquivo '{nome_arquivo}' não encontrado em /entrevistas.")

    base = video.stem
    print(f"\n=== PROCESSANDO: {base} ===")
    outdir = Path("resultados") / base
    outdir.mkdir(parents=True, exist_ok=True)

    wav = outdir / f"{base}.wav"
    json_freq = outdir / f"{base}.json"
    json_combinado = outdir / f"{base}_combinado.json"
    grafico = outdir / f"{base}_emocoes.png"

    try:
        converter_para_wav(video, wav)
        print(f"Analisando frases: {base}")
        frases = transcrever_com_tempo(str(wav))
        print(f"Analisando emoções: {base}")
        emotions = analisar_video(str(video))

        print("Salvando resultados...")
        freq = dict(Counter([e["emocao"] for e in emotions]))
        with open(json_freq, "w", encoding="utf-8") as f:
            json.dump(freq, f, ensure_ascii=False, indent=4)

        json_emotions = str(json_freq).replace(".json", "_detalhado.json")
        with open(json_emotions, "w", encoding="utf-8") as f:
            json.dump(emotions, f, ensure_ascii=False, indent=4)

        print("Combinando dados...")
        combinados = combinar(frases, emotions)
        with open(json_combinado, "w", encoding="utf-8") as f:
            json.dump(combinados, f, ensure_ascii=False, indent=4)

        salvar_grafico(emotions, grafico)
        gerar_pdf_report(outdir, base, json_freq, json_combinado, grafico)

        print(f"✔ Concluído: {base}")

    except Exception as e:
        print(f"❌ Erro em {base}: {e}")


if __name__ == "__main__":
    processar_video_unico("Gustavo Santos El Dib.mp4")
