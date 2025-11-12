import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

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

    # ... (código anterior onde o df é criado) ...

    # --- NOVO GRÁFICO 1: Heatmap (Mapa de Calor) ---
    plt.figure(figsize=(10, 8))
    
    # 'annot=True' escreve o número dentro do quadrado
    # 'cmap="YlGnBu"' define a cor (Amarelo -> Verde -> Azul)
    # 'fmt=".1f"' formata o número com 1 casa decimal
    sns.heatmap(df, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=.5)
    
    plt.title('Intensidade das Emoções por Pessoa (%)', fontsize=16)
    plt.tight_layout()
    plt.savefig(base_path / "heatmap_emocoes.png", dpi=300)
    print(f"✅ Heatmap salvo: {base_path / 'heatmap_emocoes.png'}")

    # --- NOVO GRÁFICO 2: Boxplot (Distribuição Estatística) ---
    plt.figure(figsize=(10, 6))
    
    # O boxplot ignora as pessoas e foca nas Emoções
    sns.boxplot(data=df, palette="Set3")
    
    # Adiciona os pontos individuais (swarmplot) por cima para ver onde cada pessoa cai
    sns.swarmplot(data=df, color=".25", size=5)
    
    plt.title('Variação de Cada Emoção no Grupo', fontsize=16)
    plt.ylabel('Frequência (%)')
    plt.grid(True, axis='y', alpha=0.3) # Linhas de grade ajudam a ler
    
    plt.tight_layout()
    plt.savefig(base_path / "boxplot_emocoes.png", dpi=300)
    print(f"✅ Boxplot salvo: {base_path / 'boxplot_emocoes.png'}")

if __name__ == "__main__":
    processar_emocoes("./resultados")