import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =================================================================
# 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS (REPETIÇÃO NECESSÁRIA)
# =================================================================
df_emotions = pd.read_csv("resultados/resumo_frequencias.csv", sep=";")
df_survey = pd.read_csv("Saúde Mental na EACH (respostas) - Respostas ao formulário 1.csv")

def clean_name(name):
    if pd.isna(name): return None
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

# Merge preparation
df_survey.rename(columns={'Nome completo:': 'Nome'}, inplace=True)
df_survey['Nome_cleaned'] = df_survey['Nome'].apply(clean_name)
df_emotions['Pessoa_cleaned'] = df_emotions['Pessoa'].apply(clean_name)

# Colunas base para a Regressão
reg_cols_base = ['PSS_Total', 'Idade:', 'Nome_cleaned']
# Colunas adicionais a serem incluídas no modelo estendido (para serem transformadas em dummy)
new_reg_cols = ['Sexo:', 'Você já fez/faz acompanhamento terapêutico?', 'Você já foi diagnosticado com algum transtorno de saúde mental?']

# Merge df_survey (PSS, Idade, Novas Variáveis) and df_emotions (Frequencies)
emotion_cols = ['sad', 'fear', 'surprise', 'happy', 'angry', 'disgust', 'Pessoa_cleaned']

df_reg_emotions = pd.merge(
    df_survey[reg_cols_base + new_reg_cols],
    df_emotions[emotion_cols],
    left_on='Nome_cleaned',
    right_on='Pessoa_cleaned',
    how='inner'
)

# Renomear e limpar colunas
col_map = {
    'Idade:': 'Idade', 
    'Sexo:': 'Sexo', 
    'Você já fez/faz acompanhamento terapêutico?': 'Terapia', 
    'Você já foi diagnosticado com algum transtorno de saúde mental?': 'Diagnostico'
}
df_reg_emotions.rename(columns=col_map, inplace=True)
df_reg_emotions['Idade'] = pd.to_numeric(df_reg_emotions['Idade'], errors='coerce')

# Remover NaNs (Crucial)
df_reg_emotions.dropna(inplace=True)


# =================================================================
# 2. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) NAS EMOÇÕES
# =================================================================
emotion_features = ['sad', 'fear', 'surprise', 'happy', 'angry', 'disgust']
X_emotions = df_reg_emotions[emotion_features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_emotions)

# Manter 5 componentes (conforme solicitado)
n_components_to_keep = 5
pca_final = PCA(n_components=n_components_to_keep)
X_pca = pca_final.fit_transform(X_scaled)

# Adicionar os componentes principais ao DataFrame
for i in range(n_components_to_keep):
    df_reg_emotions[f'PC{i+1}'] = X_pca[:, i]


# =================================================================
# 3. REGRESSÃO LINEAR COM VARIÁVEIS ADICIONAIS
# =================================================================

# Fórmula estendida: PC2, PC3, PC4, PC5 (do usuário) + Idade + Sexo + Terapia + Diagnostico
formula_extended = 'PSS_Total ~ PC2 + C(Terapia) + C(Diagnostico)'

print("----------------------------------------------------------")
print(f"MODELO ESTENDIDO: N = {len(df_reg_emotions)} observações.")
print("----------------------------------------------------------")

# Para um N=12, este modelo tem 1 (Intercept) + 1 (Idade) + 4 (PCs) + 1 (Sexo) + 1 (Terapia) + 1 (Diagnostico) = 9 parâmetros.
# Df Residuals = 12 - 9 = 3. O modelo é matematicamente possível, mas altamente instável.

try:
    # Usando Erros Padrão Robustos (HC0) para maior confiabilidade, como na etapa anterior.
    model_extended = smf.ols(formula_extended, data=df_reg_emotions)
    results_extended = model_extended.fit(cov_type='HC0')

    print(f"\nRESUMO DA REGRESSÃO LINEAR ESTENDIDA ({formula_extended})")
    print("----------------------------------------------------------")
    print(results_extended.summary())

except Exception as e:
    print(f"Erro ao executar a regressão OLS estendida: {e}")
    print("\nO modelo falhou devido à multicolinearidade perfeita ou N muito pequeno.")