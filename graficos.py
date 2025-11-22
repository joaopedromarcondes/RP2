import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df_freq = pd.read_csv("resultados/resumo_frequencias.csv", sep=";")
df_survey = pd.read_csv("Saúde Mental na EACH (respostas) - Respostas ao formulário 1.csv")

# Clean and Merge
df_freq['clean_name'] = df_freq['Pessoa'].str.lower().str.strip()
df_survey['clean_name'] = df_survey['Nome completo:'].str.lower().str.strip()
df_merged = pd.merge(df_freq, df_survey, on='clean_name', how='inner')

# Calculate PSS
pss_cols = df_survey.columns[16:26]
pos_indices = [3, 4, 6, 7] # Indices within the pss_cols list to reverse

df_merged['PSS_Score'] = 0
for i, col in enumerate(pss_cols):
    if i in pos_indices:
        df_merged['PSS_Score'] += (4 - df_merged[col])
    else:
        df_merged['PSS_Score'] += df_merged[col]

# Set up the plotting style
sns.set_theme(style="whitegrid")

# 1. PSS vs Emotions (Scatter Plots)
# We'll pick the top 3 most interesting emotions: Sad, Happy, Neutral
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

sns.scatterplot(data=df_merged, x='sad', y='PSS_Score', ax=axes[0], color='blue', s=100)
sns.regplot(data=df_merged, x='sad', y='PSS_Score', ax=axes[0], scatter=False, color='blue')
axes[0].set_title('PSS vs Tristeza (Expressão Facial)')
axes[0].set_xlabel('Frequência de Tristeza (%)')
axes[0].set_ylabel('Score PSS')

sns.scatterplot(data=df_merged, x='happy', y='PSS_Score', ax=axes[1], color='green', s=100)
sns.regplot(data=df_merged, x='happy', y='PSS_Score', ax=axes[1], scatter=False, color='green')
axes[1].set_title('PSS vs Felicidade (Expressão Facial)')
axes[1].set_xlabel('Frequência de Felicidade (%)')
axes[1].set_ylabel('Score PSS')

sns.scatterplot(data=df_merged, x='neutral', y='PSS_Score', ax=axes[2], color='gray', s=100)
sns.regplot(data=df_merged, x='neutral', y='PSS_Score', ax=axes[2], scatter=False, color='gray')
axes[2].set_title('PSS vs Neutralidade (Expressão Facial)')
axes[2].set_xlabel('Frequência de Neutralidade (%)')
axes[2].set_ylabel('Score PSS')

plt.tight_layout()

plt.show()

# 2. PSS vs Categorical Variables
# We will create a 2x2 grid for demographics/habits
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# PSS by Sex
sns.boxplot(data=df_merged, x='Sexo:', y='PSS_Score', ax=axes[0, 0], palette="pastel")
sns.swarmplot(data=df_merged, x='Sexo:', y='PSS_Score', ax=axes[0, 0], color=".25")
axes[0, 0].set_title('PSS por Sexo')
axes[0, 0].set_xlabel('')

# PSS by Therapy
sns.boxplot(data=df_merged, x='Você já fez/faz acompanhamento terapêutico?', y='PSS_Score', ax=axes[0, 1], palette="pastel")
sns.swarmplot(data=df_merged, x='Você já fez/faz acompanhamento terapêutico?', y='PSS_Score', ax=axes[0, 1], color=".25")
axes[0, 1].set_title('PSS por Acompanhamento Terapêutico')
axes[0, 1].set_xlabel('')
axes[0, 1].tick_params(axis='x', rotation=15)

# PSS by Sleep
# Order the sleep categories if possible, but they are strings. Let's see the unique values.
# "Entre 5 e 6 horas", "Entre 6 e 7 horas", "Entre 7 e 8 horas", "Menos de 5 horas"
sleep_order = ['Menos de 5 horas', 'Entre 5 e 6 horas', 'Entre 6 e 7 horas', 'Entre 7 e 8 horas', 'Mais de 8 horas']
# Filter order list to only include what's in the data
existing_sleep = df_merged['Em média, quantas horas de sono você tem por noite?'].unique()
sleep_order = [x for x in sleep_order if x in existing_sleep]

sns.boxplot(data=df_merged, x='Em média, quantas horas de sono você tem por noite?', y='PSS_Score', ax=axes[1, 0], order=sleep_order, palette="pastel")
sns.swarmplot(data=df_merged, x='Em média, quantas horas de sono você tem por noite?', y='PSS_Score', ax=axes[1, 0], order=sleep_order, color=".25")
axes[1, 0].set_title('PSS por Horas de Sono')
axes[1, 0].set_xlabel('')
axes[1, 0].tick_params(axis='x', rotation=15)

# PSS by Physical Activity
activity_col = 'Quantas vezes você realiza atividade física na semana?'
activity_order = ['Nenhuma', 'entre 1 e 2 vezes', 'entre 3 e 5 vezes', 'mais de 5 vezes']
existing_activity = df_merged[activity_col].unique()
activity_order = [x for x in activity_order if x in existing_activity]

sns.boxplot(data=df_merged, x=activity_col, y='PSS_Score', ax=axes[1, 1], order=activity_order, palette="pastel")
sns.swarmplot(data=df_merged, x=activity_col, y='PSS_Score', ax=axes[1, 1], order=activity_order, color=".25")
axes[1, 1].set_title('PSS por Atividade Física')
axes[1, 1].set_xlabel('')
axes[1, 1].tick_params(axis='x', rotation=15)

plt.tight_layout()
plt.show()










# Descriptive Statistics
stats_desc = df_merged['PSS_Score'].describe()

# Plotting
sns.set_theme(style="whitegrid", context="talk")

# Create a figure with a layout that allows a boxplot on top of a histogram
fig, (ax_box, ax_hist) = plt.subplots(2, 1, sharex=True, 
                                    gridspec_kw={"height_ratios": (.15, .85)}, 
                                    figsize=(10, 7))

# Boxplot
sns.boxplot(x=df_merged['PSS_Score'], ax=ax_box, color="#3498db")
ax_box.set(xlabel='') # Remove x label for the boxplot

# Histogram with KDE
sns.histplot(df_merged['PSS_Score'], ax=ax_hist, kde=True, color="#3498db", bins=8, edgecolor='white')

# Add mean line
mean_val = df_merged['PSS_Score'].mean()
ax_hist.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Média: {mean_val:.1f}')

# Add labels and title
ax_hist.set_xlabel('Score de Estresse Percebido (PSS)', fontsize=12)
ax_hist.set_ylabel('Frequência', fontsize=12)
ax_hist.legend()

# Title for the whole figure
fig.suptitle('Distribuição Descritiva do PSS', fontsize=16, weight='bold')

sns.despine(ax=ax_hist)
sns.despine(ax=ax_box, left=True)

plt.tight_layout()
plt.show()

print("\nDescriptive Statistics for PSS Score:\n", stats_desc)