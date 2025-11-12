import subprocess
import importlib
from pathlib import Path
from typing import Callable

from analise_de_sentimentos import analise_sentimentos

# /home/jp/Documentos/USP/6sem/RP2/ideias_iniciais/outra_analise.py

MP4_PATH = Path("teste_inicial/saida.mp4")
WAV_PATH = MP4_PATH.with_suffix(".wav")


def converter_mp4_para_wav(mp4: Path, wav: Path) -> None:
    mp4 = Path(mp4)
    wav = Path(wav)
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(mp4),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(wav),
    ]
    subprocess.run(cmd, check=True)



def main():
    if not MP4_PATH.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {MP4_PATH}")
    converter_mp4_para_wav(MP4_PATH, WAV_PATH)

    analise_sentimentos(str(WAV_PATH), str(MP4_PATH))



if __name__ == "__main__":
    main()