import pandas as pd
import numpy as np

# =================================================================
# 1. CARREGAMENTO E INSPEÇÃO DOS DADOS
# =================================================================

# Carrega o arquivo de frequências de emoções (separador: ';')
df_emotions = pd.read_csv("resultados/resumo_frequencias.csv", sep=";")

# Carrega o arquivo de respostas do questionário (separador padrão: ',')
df_survey = pd.read_csv("Saúde Mental na EACH (respostas) - Respostas ao formulário 1.csv")

# =================================================================
# 2. LIMPEZA DOS NOMES PARA JUNÇÃO (MERGE)
# =================================================================

def clean_name(name):
    """Limpa o nome para facilitar a junção (lower case e strip)."""
    return name.lower().strip()

# Renomeia e limpa a coluna de nomes no df_survey
df_survey.rename(columns={'Nome completo:': 'Nome'}, inplace=True)
df_survey['Nome_cleaned'] = df_survey['Nome'].apply(clean_name)

# Limpa a coluna de pessoas no df_emotions
df_emotions['Pessoa_cleaned'] = df_emotions['Pessoa'].apply(clean_name)

# =================================================================
# 3. CÁLCULO DA PONTUAÇÃO PSS TOTAL
# =================================================================

# Identifica as 10 colunas PSS (assumindo que são as últimas 10)
pss_cols = df_survey.columns[-10:].tolist()

# Função de Escoragem Reversa (Para escala 0-4: 0->4, 1->3, 2->2, 3->1, 4->0)
def reverse_score(series):
    return 4 - series

# DataFrame temporário para o cálculo do PSS
# --- Solução para o PSS Total ---

# 1. Identificação das 10 colunas PSS
pss_cols = df_survey.columns[-10:].tolist()

# 2. Criação do DataFrame temporário df_pss com conversão forçada
df_pss = df_survey[pss_cols].copy()

# Tenta converter todas as colunas PSS para inteiro.
# 'errors="coerce"' substitui quaisquer valores não-numéricos (texto, espaços) por NaN (Not a Number).
# É crucial fazer isso antes de somar.
for col in pss_cols:
    df_pss[col] = pd.to_numeric(df_pss[col], errors='coerce').astype('Int64') # Int64 maiúsculo suporta NaN

# 3. Definição da Função de Escoragem Reversa (Para escala 0-4: 0->4, 3->1, etc.)
def reverse_score(series):
    # O PSS vai ser somado, então podemos simplesmente usar a lógica
    return 4 - series

# 4. Aplicação do Escoramento Reverso (Itens PSS 4, 5, 7, 8 - índices 3, 4, 6, 7)
reverse_indices = [3, 4, 6, 7] 

for idx in reverse_indices:
    col_name = pss_cols[idx]
    # Aplica a escoragem reversa
    df_pss[col_name] = reverse_score(df_pss[col_name])

# 5. Cálculo do PSS Total (Com 'skipna=True' para ignorar os NaN se houver algum erro de digitação/dado faltante)
df_survey['PSS_Total'] = df_pss.sum(axis=1, skipna=True)

# Índices das colunas PSS a serem revertidas (4, 5, 7, 8 - são os índices 3, 4, 6, 7 da lista pss_cols)
reverse_indices = [3, 4, 6, 7] 

for idx in reverse_indices:
    col_name = pss_cols[idx]
    # Aplica 4 - score para inverter a pontuação
    df_pss[col_name] = reverse_score(df_pss[col_name])

# Cálculo do PSS Total (Soma das 10 pontuações ajustadas)
df_survey['PSS_Total'] = df_pss.sum(axis=1)

# =================================================================
# 4. DETERMINAÇÃO DA EMOÇÃO PREDOMINANTE
# =================================================================

emotion_cols = ['sad', 'neutral', 'fear', 'surprise', 'happy', 'angry', 'disgust']

# idxmax(axis=1) encontra o rótulo da coluna com o valor máximo em cada linha
df_emotions['Predominant_Emotion'] = df_emotions[emotion_cols].idxmax(axis=1)

# =================================================================
# 5. JUNÇÃO DOS DATAFRAMES E ANÁLISE FINAL
# =================================================================

# Junção dos dados PSS e Emoções pelo nome limpo
df_analysis = pd.merge(
    df_survey,
    df_emotions[['Pessoa_cleaned', 'Predominant_Emotion']],
    left_on='Nome_cleaned',
    right_on='Pessoa_cleaned',
    how='inner'
)

# Análise: PSS Médio por Emoção Predominante
pss_by_emotion = df_analysis.groupby('Predominant_Emotion')['PSS_Total'].agg(['mean', 'std', 'count']).sort_values(by='mean', ascending=False)

print("--------------------------------------------------")
print("RESULTADO FINAL: PSS MÉDIO POR EMOÇÃO PREDOMINANTE")
print("--------------------------------------------------")
print(pss_by_emotion)