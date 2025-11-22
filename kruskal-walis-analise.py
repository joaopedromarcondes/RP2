import pandas as pd
from scipy import stats
import numpy as np

# Load data
df_freq = pd.read_csv("resultados/resumo_frequencias.csv", sep=";")
df_survey = pd.read_csv("Saúde Mental na EACH (respostas) - Respostas ao formulário 1.csv")

# Clean and Merge
df_freq['clean_name'] = df_freq['Pessoa'].str.lower().str.strip()
df_survey['clean_name'] = df_survey['Nome completo:'].str.lower().str.strip()
df_merged = pd.merge(df_freq, df_survey, on='clean_name', how='inner')

# Calculate PSS
pss_cols = df_survey.columns[16:26]
pos_indices = [3, 4, 6, 7] # Indices for items 4, 5, 7, 8 (0-indexed: 3, 4, 6, 7)

df_merged['PSS_Score'] = 0
for i, col in enumerate(pss_cols):
    if i in pos_indices:
        df_merged['PSS_Score'] += (4 - df_merged[col])
    else:
        df_merged['PSS_Score'] += df_merged[col]

# Inspect groups
print("Unique Sleep Values:", df_merged['Em média, quantas horas de sono você tem por noite?'].unique())
print("Unique Therapy Values:", df_merged['Você já fez/faz acompanhamento terapêutico?'].unique())

# --- Statistical Tests ---

# 1. Test for Sleep (Kruskal-Wallis)
# Group data by sleep
sleep_groups = []
for sleep_val in df_merged['Em média, quantas horas de sono você tem por noite?'].unique():
    sleep_groups.append(df_merged[df_merged['Em média, quantas horas de sono você tem por noite?'] == sleep_val]['PSS_Score'])

# Perform Kruskal-Wallis
stat_sleep, p_sleep = stats.kruskal(*sleep_groups)

# 2. Test for Therapy (Mann-Whitney or Kruskal-Wallis depending on group count)
therapy_counts = df_merged['Você já fez/faz acompanhamento terapêutico?'].value_counts()
print("\nTherapy Counts:\n", therapy_counts)

therapy_groups = []
for therapy_val in df_merged['Você já fez/faz acompanhamento terapêutico?'].unique():
    therapy_groups.append(df_merged[df_merged['Você já fez/faz acompanhamento terapêutico?'] == therapy_val]['PSS_Score'])

stat_therapy, p_therapy = stats.kruskal(*therapy_groups)

# 3. Categorizing PSS
# Scale: 0-13 (Low), 14-26 (Moderate), 27-40 (High)
def categorize_pss(score):
    if score <= 13:
        return 'Baixo'
    elif score <= 26:
        return 'Moderado'
    else:
        return 'Alto'

df_merged['PSS_Category'] = df_merged['PSS_Score'].apply(categorize_pss)
print("\nPSS Categories Distribution:\n", df_merged['PSS_Category'].value_counts())

# Output results
print(f"\nKruskal-Wallis Test for Sleep: Statistic={stat_sleep:.3f}, p-value={p_sleep:.3f}")
print(f"Kruskal-Wallis Test for Therapy: Statistic={stat_therapy:.3f}, p-value={p_therapy:.3f}")