import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def processar_emocoes(caminho_base):
    base_path = Path(caminho_base)
    dados_gerais = []
    
    print(f"🔍 Procurando arquivos em: {base_path.resolve()}")

    for pasta_pessoa in base_path.iterdir():
        if pasta_pessoa.is_dir():
            # --- CORREÇÃO 1: Limpeza do Nome ---
            nome_bruto = pasta_pessoa.name
            # Troca underline/hífen por espaço e coloca Iniciais Maiúsculas
            # Ex: "joao_silva" vira "Joao Silva"
            nome_bonito = nome_bruto.replace("_", " ").replace("-", " ").title()
            
            # Procura o JSON (usando o nome original da pasta)
            arquivo_json = pasta_pessoa / f"{nome_bruto}.json"
            
            if arquivo_json.exists():
                try:
                    with open(arquivo_json, 'r', encoding='utf-8') as f:
                        contagem_absoluta = json.load(f)
                    
                    total_emocoes = sum(contagem_absoluta.values())
                    
                    if total_emocoes > 0:
                        # Usa o 'nome_bonito' para o gráfico
                        dados_pessoa = {'Pessoa': nome_bonito}
                        
                        for emocao, valor in contagem_absoluta.items():
                            freq_relativa = (valor / total_emocoes) * 100
                            dados_pessoa[emocao] = freq_relativa
                        
                        dados_gerais.append(dados_pessoa)
                        
                except Exception as e:
                    print(f"❌ Erro ao ler {arquivo_json}: {e}")

    if not dados_gerais:
        print("Nenhum dado encontrado.")
        return

    df = pd.DataFrame(dados_gerais)
    df = df.fillna(0)
    df.set_index('Pessoa', inplace=True)

    # Salvar CSV/JSON
    df_final = df.round(2)
    df_final.to_csv(base_path / "resumo_frequencias.csv", encoding='utf-8-sig', sep=';')
    df_final.to_json(base_path / "resumo_frequencias.json", orient='index', indent=4)

    # --- GERAÇÃO DO GRÁFICO MELHORADA ---
    print("📊 Gerando gráfico...")
    
    # Aumentei um pouco a figura para caber os nomes
    plt.figure(figsize=(14, 8)) 
    
    ax = df.plot(kind='bar', stacked=True, colormap='Spectral', figsize=(12, 7))

    plt.title('Distribuição de Emoções (Frequência Relativa)', fontsize=16)
    plt.ylabel('Porcentagem (%)', fontsize=12)
    plt.xlabel('Pessoa', fontsize=12)
    
    # --- CORREÇÃO 2: Rotação e Alinhamento ---
    # rotation=45: Inclina o texto
    # ha='right': Alinha o final da palavra com o tracinho do eixo (fica muito mais legível)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    plt.legend(title='Emoções', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Garante que nada seja cortado na imagem final
    plt.tight_layout()

    caminho_grafico = base_path / "comparativo_emocoes.png"
    plt.savefig(caminho_grafico, dpi=300)
    print(f"✅ Gráfico salvo: {caminho_grafico}")

if __name__ == "__main__":
    processar_emocoes("./resultados")