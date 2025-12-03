import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

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
emotion_cols_without_neutral = ['sad', 'fear', 'surprise', 'happy', 'angry', 'disgust', 'neutral']
df_emotions['Predominant_Emotion'] = df_emotions[emotion_cols_without_neutral].idxmax(axis=1)

# Merge
df_survey.rename(columns={'Nome completo:': 'Nome'}, inplace=True)
df_survey['Nome_cleaned'] = df_survey['Nome'].apply(clean_name)
df_emotions['Pessoa_cleaned'] = df_emotions['Pessoa'].apply(clean_name)

df_analysis = pd.merge(
    df_survey,
    df_emotions[['Pessoa_cleaned', 'sad', 'fear', 'surprise', 'happy', 'angry', 'disgust', 'neutral']],
    left_on='Nome_cleaned',
    right_on='Pessoa_cleaned',
    how='inner'
)

df_analysis.drop(['Pessoa_cleaned', 'Carimbo de data/hora', 'Endereço de e-mail', 'Nome', 'No último mês, com que frequência você ficou chateado(a) por algo que aconteceu inesperadamente?',
       'No último mês, com que frequência você sentiu que não conseguia controlar as coisas importantes na sua vida?',
       'No último mês, com que frequência você se sentiu nervoso(a) e estressado(a)?',
       'No último mês, com que frequência você se sentiu confiante sobre sua capacidade de lidar com seus problemas pessoais?',
       'No último mês, com que frequência você sentiu que as coisas estavam indo do seu jeito?',
       'No último mês, com que frequência você descobriu que não conseguia lidar com todas as coisas que tinha para fazer?',
       'No último mês, com que frequência você conseguiu controlar as irritações em sua vida?',
       'No último mês, com que frequência você sentiu que estava no controle das coisas?',
       'No último mês, com que frequência você ficou com raiva por causa de coisas que aconteceram e estavam fora do seu controle?',
       'No último mês, com que frequência você sentiu que as dificuldades estavam se acumulando tanto que você não conseguia superá-las?',
       'Se sim, qual?', 
       'Você faz uso de algum medicamento? Se sim, qual?'], axis=1, inplace=True)

df_analysis.rename(columns={
    'sad': 'Tristeza',
    'fear': 'Medo',
    'surprise': 'Surpresa',
    'happy': 'Felicidade',
    'angry': 'Raiva',
    'disgust': 'Nojo',
    'neutral': 'Neutralidade'
}, inplace=True)

df_analysis.rename(columns={'Idade:': 'Idade', 'Sexo:': 'Sexo',
       'Você já fez/faz acompanhamento terapêutico?': 'Acompanhamento_Terapeutico',
       'Você já foi diagnosticado com algum transtorno de saúde mental?': 'Diagnosticado',
       'Em média, quantas horas de sono você tem por noite?': "Horas_Sono",
       'Quantas vezes você realiza atividade física na semana?': "Atividade_Fisica",
       'Quanto tempo você dedica para atividades de lazer?': "Atividade_Lazer",
       "Qual é o seu curso?": "Curso",
       "Semestre atual:": "Semestre Atual",
       "Renda familiar:": "Renda Familiar",
       'Qual é a raça/etnia que você se identifica?': "Etnia"}, inplace=True)

print(df_analysis.head())
print("\nColumns in Merged DataFrame:\n", df_analysis.columns.tolist())


df_analysis["Emoções_Positivas"] = df_analysis["Felicidade"] + df_analysis["Surpresa"]
df_analysis["Emoções Negativas"] = df_analysis["Tristeza"] + df_analysis["Medo"] + df_analysis["Raiva"] + df_analysis["Nojo"]

print("\nDataFrame after adding Positive and Negative Emotions:\n", df_analysis)

plot_data = df_analysis[['Emoções_Positivas', 'Emoções Negativas']]

# Plot
plt.figure(figsize=(8, 6))
sns.set_theme(style="whitegrid")

# Create boxplot with mean line
# showmeans=True displays the mean. meanline=True draws it as a line instead of a point.
# meanprops allows customizing the line.
ax = sns.boxplot(data=plot_data, showmeans=True, meanline=True,
                 meanprops={'color': 'red', 'ls': '--', 'linewidth': 2})

plt.title('Distribuição das Emoções (Positivas vs Negativas)', fontsize=14)
plt.ylabel('Frequência Acumulada (%)')

# Optional: Add text annotation for the mean values
means = plot_data.mean()
for i, mean_val in enumerate(means):
    ax.text(i, mean_val + 2, f'Média: {mean_val:.2f}', 
            horizontalalignment='center', color='red', weight='bold')

plt.tight_layout()
plt.show()


df_analysis["Emoção Predominante"] = np.where(
    df_analysis["Emoções_Positivas"] >= 7,
    "Positiva",
    "Negativa"
)   

plt.figure(figsize=(8, 6)) 
sns.boxplot(data=df_analysis, x='Emoção Predominante', y='PSS_Total', palette='Set2')
plt.title('Distribuição do PSS por Grupo Emocional (Positivo vs Negativo)')
plt.ylabel('Nível de Estresse (PSS)')
plt.xlabel('Tipo de Emoção Predominante no Rosto')
plt.show()


print("\nDataFrame after adding Predominant Emotion:\n", df_analysis[['Emoções_Positivas', 'Emoções Negativas', 'Emoção Predominante']].head())

""" plt.bar(df_analysis['Emoção Predominante'].value_counts().index, df_analysis['Emoção Predominante'].value_counts().values)



plt.show() """

print(df_analysis['Emoção Predominante'].value_counts())


# Teste t de Student
grupo_pos = df_analysis[df_analysis['Emoção Predominante'] == 'Positiva']['PSS_Total']
grupo_neg = df_analysis[df_analysis['Emoção Predominante'] == 'Negativa']['PSS_Total']



t_stat, p_val = stats.ttest_ind(grupo_pos, grupo_neg, equal_var=False)  # Welch t-test
print(t_stat, p_val)


""" import statsmodels.formula.api as smf

model = smf.ols('PSS_Total ~ Emoções_Positivas', data=df_analysis).fit()
print(model.summary()) """

df_analysis["Uma_Emoção"] = df_analysis['Emoções Negativas'] - df_analysis['Emoções_Positivas']

from scipy.stats import spearmanr

correlacao, p_valor = spearmanr(df_analysis['Uma_Emoção'], df_analysis['PSS_Total'])

stat, p_valor = stats.shapiro(grupo_neg)
print(f"Shapiro-Wilk: p-valor = {p_valor:.4f}")

if p_valor > 0.05:
    print("Distribuição Normal (p > 0.05)")
else:
    print("Distribuição Não-Normal (p < 0.05)")

# 2. Gráfico Q-Q (Visual)
stats.probplot(grupo_neg, dist="norm", plot=plt)
plt.show()

estatistica, p_valor = stats.mannwhitneyu(grupo_pos, grupo_neg)

print(f"Valor-p: {p_valor:.4f}")

# Interpretação
if p_valor < 0.05:
    print("Diferença estatisticamente significativa!")
else:
    print("Não há diferença significativa entre os grupos.")


print("Correlação de Spearman:", correlacao)
print("p-valor:", p_valor)

# --- Plotting ---
sns.set_theme(style="whitegrid", context="talk") 

plt.figure(figsize=(10, 7))

# Simple scatter plot without categorical distinction
sns.scatterplot(
    data=df_analysis,
    x='Emoções_Positivas',
    y='Emoções Negativas',
    color='#3498db', # Nice flat blue
    s=200, # Size
    alpha=0.8,
    edgecolor='black'
)

# Regression line
sns.regplot(
    data=df_analysis,
    x='Emoções_Positivas',
    y='Emoções Negativas',
    scatter=False,
    color='gray',
    line_kws={'linestyle': '--', 'alpha': 0.5}
)

plt.title('Relação: Emoções Positivas vs. Negativas', fontsize=16, weight='bold', pad=20)
plt.xlabel('Soma de Emoções Positivas (%)', fontsize=12)
plt.ylabel('Soma de Emoções Negativas (%)', fontsize=12)

sns.despine()

# Add annotations for extreme points
max_neg = df_analysis.loc[df_analysis['Emoções Negativas'].idxmax()]


plt.tight_layout()

plt.show()

print(df_analysis[['Emoções_Positivas', 'Emoções Negativas']].describe())

print("Infos das idades:", df_analysis['Emoções Negativas'].describe())