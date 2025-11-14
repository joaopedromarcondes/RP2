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

# Merge df_survey (PSS, Idade) and df_emotions (Frequencies)
emotion_cols = ['sad', 'fear', 'surprise', 'happy', 'angry', 'disgust', 'neutral', 'Pessoa_cleaned']
df_reg_emotions = pd.merge(
    df_survey[['PSS_Total', 'Idade:', 'Nome_cleaned']],
    df_emotions[emotion_cols],
    left_on='Nome_cleaned',
    right_on='Pessoa_cleaned',
    how='inner'
)

# Renomear Idade e garantir tipo numérico
df_reg_emotions.rename(columns={'Idade:': 'Idade'}, inplace=True)
df_reg_emotions['Idade'] = pd.to_numeric(df_reg_emotions['Idade'], errors='coerce')

# Remover NaNs
df_reg_emotions.dropna(inplace=True)

#print(df_reg_emotions.head())
# =================================================================
# 2. ANÁLISE DE COMPONENTES PRINCIPAIS (PCA) NAS EMOÇÕES
# =================================================================

emotion_features = ['sad', 'fear', 'surprise', 'happy', 'angry', 'disgust']
X_emotions = df_reg_emotions[emotion_features]

# Padronização dos dados (Importante para PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_emotions)

# Aplicação do PCA
pca = PCA(n_components=6) 
pca.fit(X_scaled)

# Cálculo da variância explicada acumulada
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Escolha do número de componentes: Manter 2 componentes (PC1 e PC2)
n_components_to_keep = 5

# Transformação dos dados
pca_final = PCA(n_components=n_components_to_keep)
X_pca = pca_final.fit_transform(X_scaled)

# Adicionar os componentes principais ao DataFrame de regressão
for i in range(n_components_to_keep):
    df_reg_emotions[f'PC{i+1}'] = X_pca[:, i]

print("----------------------------------------------------------")
print("PCA: Variância Explicada Acumulada")
print("----------------------------------------------------------")
for i, var in enumerate(cumulative_explained_variance):
    print(f"Componentes 1 até {i+1} explicam: {var:.4f}")

# =================================================================
# 3. REGRESSÃO LINEAR COM COMPONENTES PRINCIPAIS
# =================================================================

# Fórmula: PSS_Total ~ Idade + PC1 + PC2
formula_pca = 'PSS_Total ~ Idade + PC2 + PC3 + PC4 + PC5'

print(f"\nNúmero de observações restantes para o modelo: N = {len(df_reg_emotions)}")

try:
    # OLS: Ordinary Least Squares (Mínimos Quadrados Ordinários)
    model_pca = smf.ols(formula_pca, data=df_reg_emotions)
    results_pca = model_pca.fit(cov_type='fixed scale')  

    print("\n----------------------------------------------------------")
    print(f"RESUMO DA REGRESSÃO LINEAR ({formula_pca})")
    print("----------------------------------------------------------")
    print(results_pca.summary())

except Exception as e:
    print(f"Erro ao executar a regressão OLS com PCA: {e}")