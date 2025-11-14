import pandas as pd
import numpy as np

# =================================================================
# 1. CARREGAMENTO DOS DADOS
# =================================================================
df_emotions = pd.read_csv("resultados/resumo_frequencias.csv", sep=";")
df_survey = pd.read_csv("Saúde Mental na EACH (respostas) - Respostas ao formulário 1.csv")

# =================================================================
# 2. LIMPEZA E CÁLCULO PSS TOTAL
# =================================================================
def clean_name(name):
    if pd.isna(name):
        return None
    return str(name).lower().strip()

# PSS Total Calculation
pss_cols = df_survey.columns[-10:].tolist()
def reverse_score(series): return 4 - series
df_pss = df_survey[pss_cols].copy()
for col in pss_cols:
    df_pss[col] = pd.to_numeric(df_pss[col], errors='coerce').astype('Int64') 
reverse_indices = [3, 4, 6, 7] 
for idx in reverse_indices:
    if idx < len(pss_cols):
        col_name = pss_cols[idx]
        df_pss[col_name] = reverse_score(df_pss[col_name])
df_survey['PSS_Total'] = df_pss.sum(axis=1, skipna=True)

# Emotion Predominant Calculation (without neutral)
emotion_cols_without_neutral = ['sad', 'fear', 'surprise', 'happy', 'angry', 'disgust']
df_emotions['Predominant_Emotion'] = df_emotions[emotion_cols_without_neutral].idxmax(axis=1)

# Merge
df_survey.rename(columns={'Nome completo:': 'Nome'}, inplace=True)
df_survey['Nome_cleaned'] = df_survey['Nome'].apply(clean_name)
df_emotions['Pessoa_cleaned'] = df_emotions['Pessoa'].apply(clean_name)

df_analysis = pd.merge(
    df_survey,
    df_emotions[['Pessoa_cleaned', 'Predominant_Emotion']],
    left_on='Nome_cleaned',
    right_on='Pessoa_cleaned',
    how='inner'
)

# =================================================================
# 3. NOVAS CATEGORIZAÇÕES
# =================================================================

# 3.1. Agrupamento das Emoções
positive_emotions = ['happy', 'surprise']
negative_emotions = ['sad', 'fear', 'angry', 'disgust'] 

def group_emotion(emotion):
    if emotion in positive_emotions:
        return 'Positiva'
    elif emotion in negative_emotions:
        return 'Negativa'
    else:
        return 'Outra/Não Classificada'

df_analysis['Emotion_Group'] = df_analysis['Predominant_Emotion'].apply(group_emotion)


# 3.2. Categorização PSS (3 níveis: Baixo, Moderado, Alto)
bins_pss = [0, 13, 26, 40]
labels_pss = ['Baixo (0-13)', 'Moderado (14-26)', 'Alto (27-40)']

df_analysis['PSS_Level_3'] = pd.cut(
    df_analysis['PSS_Total'], 
    bins=bins_pss, 
    labels=labels_pss, 
    right=True, 
    include_lowest=True
)

# =================================================================
# 4. NOVA TABELA DE CONTINGÊNCIA
# =================================================================

contingency_table_grouped = pd.crosstab(
    df_analysis['Emotion_Group'],
    df_analysis['PSS_Level_3']
)

print("--- TABELA DE CONTINGÊNCIA FINAL ---")
print("Relação: Grupo de Emoção (Positiva/Negativa) vs Nível de Estresse PSS (3 Níveis)")
print(contingency_table_grouped)


from scipy.stats import chi2_contingency

chi2, p, dof, expected = chi2_contingency(contingency_table_grouped)

# Imprime os resultados
print(f"Estatística Qui-Quadrado (Chi2): {chi2}")
print(f"Valor p (p-value): {p}")
print(f"Graus de Liberdade (df): {dof}")
print("Frequências Esperadas (Expected Frequencies - E):", expected)