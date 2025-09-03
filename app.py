import sounddevice as sd
import whisper
import datetime
import json
import numpy as np
import tempfile
from scipy.io.wavfile import write
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

# ---- Configuração ----
historico = []

# ---- LLM (Ollama) ----
llm = Ollama(model="llama3.2")

# Prompt que instrui o modelo a ser entrevistador
template = PromptTemplate(
    input_variables=["historico"],
    template=(
        "Você é um entrevistador em uma conversa. "
        "Aqui está o histórico da entrevista até agora:\n\n{historico}\n\n"
        "Com base nisso, faça a próxima pergunta de forma natural, "
        "em português, SEM responder por você mesmo. "
        "Apenas gere a nova pergunta."
    )
)

# ---- Carregar modelo Whisper ----
modelo_whisper = whisper.load_model("base")

# ---- Função para capturar fala ----
def gravar_resposta(duracao=5, fs=16000):
    print("🎤 Gravando... fale agora!")
    audio = sd.rec(int(duracao * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    print("✅ Gravação concluída.")
    return (fs, audio)

# ---- Função para transcrever com Whisper ----
def transcrever(audio_data, fs):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, fs, (audio_data * 32767).astype(np.int16))  # salvar como WAV
        result = modelo_whisper.transcribe(tmpfile.name, language="pt", fp16=False)
    return result["text"]

# ---- Entrevista ----
# Primeira pergunta fixa
pergunta = "Qual é o seu nome?"

for i in range(5):  # define quantas perguntas quer no total
    print("\n🤖 Entrevistador:", pergunta)
    ts_pergunta = datetime.datetime.now().isoformat()

    # Gravar áudio do usuário
    fs, audio = gravar_resposta(5)
    resposta = transcrever(audio, fs)
    print("👤 Você:", resposta)

    ts_resposta = datetime.datetime.now().isoformat()

    # Registrar histórico
    historico.append({
        "pergunta": pergunta,
        "pergunta_timestamp": ts_pergunta,
        "resposta": resposta,
        "resposta_timestamp": ts_resposta
    })

    # Gerar próxima pergunta pelo modelo
    contexto = "\n".join(
        [f"Q: {h['pergunta']}\nA: {h['resposta']}" for h in historico]
    )
    pergunta = llm(template.format(historico=contexto)).strip()

# ---- Salvar resultado final ----
with open("entrevista.json", "w", encoding="utf-8") as f:
    json.dump(historico, f, indent=4, ensure_ascii=False)

print("\n📁 Entrevista salva em entrevista.json")
