import sounddevice as sd
import whisper
import datetime
import json
import numpy as np
import tempfile
from scipy.io.wavfile import write
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import tkinter as tk
from tkinter import messagebox

# ---- Configuração ----
historico = []
llm = Ollama(model="llama3.2")
template = PromptTemplate(
    input_variables=["historico", "pergunta_nova"],
    template=(
        "Você é um entrevistador em uma conversa sobre saúde mental. "
        "Aqui está o histórico da entrevista até agora:\n\n{historico}\n\n"
        "Essa é a próxima pergunta que você deverá fazer:\n\n{pergunta_nova}\n\n."
        "Com base nisso, faça alterações na pergunta de forma natural, "
        "em português, SEM responder por você mesmo. "
        "Não quero sugestões de perguntas, quero que você apenas reformule a pergunta dada,"
        "com base no histórico.\n"
    )
)
modelo_whisper = whisper.load_model("base")

perguntas = [
    "Qual é o seu nome?",
    "O que você acha da saúde mental na EACH-USP?",
    "Como você está se sentindo hoje?",
    "Você tem enfrentado algum desafio recentemente?",
    "O que você faz para cuidar da sua saúde mental?",
    "Você já procurou ajuda profissional para sua saúde mental?",
    "Como você lida com o estresse no seu dia a dia?",
]
num_pergunta_atual = 0

# ---- Variáveis de gravação ----
fs = 16000
duracao = 5
audio_data = None
gravando = False

def iniciar_gravacao():
    global gravando, audio_data
    gravando = True
    status_label.config(text="🎤 Gravando... fale agora!")
    audio_data = sd.rec(int(duracao * fs), samplerate=fs, channels=1, dtype="float32")
    # Não espera aqui! O usuário vai clicar em "Parar" quando quiser.

def parar_gravacao():
    global gravando, audio_data
    if gravando and audio_data is not None:
        sd.wait()  # Finaliza a gravação
        gravando = False
        status_label.config(text="✅ Gravação concluída.")
        resposta = transcrever(audio_data, fs)
        resposta_label.config(text=f"👤 Você: {resposta}")
        registrar_resposta(resposta)
        audio_data = None
    else:
        messagebox.showinfo("Info", "Nenhuma gravação em andamento.")

def transcrever(audio_data, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, fs, (audio_data * 32767).astype(np.int16))
        result = modelo_whisper.transcribe(tmpfile.name, language="pt", fp16=False)
        print(result["text"])
    return result["text"]

def registrar_resposta(resposta):
    global pergunta, perguntas, num_pergunta_atual
    ts_pergunta = datetime.datetime.now().isoformat()
    ts_resposta = datetime.datetime.now().isoformat()
    contexto = "\n".join(
        [f"Q: {h['pergunta']}\nA: {h['resposta']}" for h in historico]
    )
    pergunta_nova = llm(template.format(historico=contexto, pergunta_nova=perguntas[num_pergunta_atual]), ).strip()
    historico.append({
        "pergunta": pergunta,
        "pergunta_timestamp": ts_pergunta,
        "resposta": resposta,
        "resposta_timestamp": ts_resposta,
        "proxima_pergunta": pergunta_nova
    })
    num_pergunta_atual += 1
    pergunta_label.config(text=f"🤖 Entrevistador: {pergunta_nova}")
    pergunta = pergunta_nova

def salvar_entrevista():
    with open("entrevista.json", "w", encoding="utf-8") as f:
        json.dump(historico, f, indent=4, ensure_ascii=False)
    messagebox.showinfo("Salvo", "📁 Entrevista salva em entrevista.json")

# ---- Interface gráfica ----
pergunta = perguntas[num_pergunta_atual]
num_pergunta_atual += 1

root = tk.Tk()
root.title("Entrevista Saúde Mental")

pergunta_label = tk.Label(
    root,
    text=f"🤖 Entrevistador: {pergunta}",
    font=("Arial", 14),
    wraplength=500,  # Limita largura e quebra linha
    justify="left"   # Alinha à esquerda
)
pergunta_label.pack(pady=10)

resposta_label = tk.Label(root, text="👤 Você: ", font=("Arial", 12))
resposta_label.pack(pady=10)

status_label = tk.Label(root, text="", font=("Arial", 10))
status_label.pack(pady=5)

btn_gravar = tk.Button(root, text="Começar Gravação", command=iniciar_gravacao)
btn_gravar.pack(side=tk.LEFT, padx=10, pady=20)

btn_parar = tk.Button(root, text="Parar Gravação", command=parar_gravacao)
btn_parar.pack(side=tk.LEFT, padx=10, pady=20)

btn_salvar = tk.Button(root, text="Salvar Entrevista", command=salvar_entrevista)
btn_salvar.pack(side=tk.RIGHT, padx=10, pady=20)

root.mainloop()
