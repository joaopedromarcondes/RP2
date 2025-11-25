import os
import cv2
import numpy as np

def analisar_videos(pasta):
    # Lista de extensões de vídeo comuns
    extensoes_validas = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
    duracoes = []

    # Verifica se a pasta existe
    if not os.path.exists(pasta):
        print(f"Erro: A pasta '{pasta}' não foi encontrada.")
        return

    print(f"Lendo vídeos na pasta: {pasta} ...")

    # Percorre todos os arquivos da pasta (incluindo subpastas)
    for raiz, _, arquivos in os.walk(pasta):
        for arquivo in arquivos:
            extensao = os.path.splitext(arquivo)[1].lower()
            
            if extensao in extensoes_validas:
                caminho_completo = os.path.join(raiz, arquivo)
                
                try:
                    # Abre o vídeo para pegar metadados
                    cap = cv2.VideoCapture(caminho_completo)
                    
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        
                        # Calcula duração em segundos (Frames / Frames por Segundo)
                        if fps > 0:
                            duracao_segundos = frame_count / fps
                            duracoes.append(duracao_segundos)
                    
                    cap.release()
                except Exception as e:
                    print(f"Não foi possível ler {arquivo}: {e}")

    if duracoes:
        media = np.mean(duracoes)
        dp = np.std(duracoes)

        # Função auxiliar para formatar o tempo
        def formatar_tempo(segundos):
            m, s = divmod(segundos, 60)
            h, m = divmod(m, 60)
            if h > 0:
                return f"{int(h)}h {int(m)}m {int(s)}s"
            return f"{int(m)}m {int(s)}s"

        print("\n" + "="*30)
        print(f"Total de vídeos analisados: {len(duracoes)}")
        print(f"Média de tempo: {formatar_tempo(media)} ({media:.2f} segundos)")
        print(f"Desvio Padrão: {formatar_tempo(dp)} ({dp:.2f} segundos)")
        print("="*30)
    else:
        print("Nenhum vídeo válido encontrado.")

# --- Execução ---
# Altere o caminho abaixo se necessário
caminho_da_pasta = './entrevistas' 
analisar_videos(caminho_da_pasta)