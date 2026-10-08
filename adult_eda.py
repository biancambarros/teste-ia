import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # 1. Carregar o arquivo CSV do link oficial
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
    
    # 2. Definir as colunas principais
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex', 
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
    ]
    
    # Lendo o CSV. Usamos skipinitialspace=True porque o dataset possui espaços após as vírgulas
    print("Carregando os dados...")
    df = pd.read_csv(url, names=columns, skipinitialspace=True)
    
    # 3. Exibir o tamanho do DataFrame e as 5 primeiras linhas
    print(f"\nTamanho do DataFrame (Linhas, Colunas): {df.shape}")
    print("\nPrimeiras 5 linhas:")
    print(df.head())
    
    # 4. Calcular a distribuição de renda separada por gênero
    print("\nDistribuição de Renda por Gênero:")
    distribuicao = pd.crosstab(df['sex'], df['income'])
    print(distribuicao)
    
    # 5. Plotar um gráfico de barras mostrando a contagem por raça, colorido por renda
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='race', hue='income', palette='Set2')
    plt.title('Distribuição de Renda por Raça')
    plt.xlabel('Raça')
    plt.ylabel('Contagem')
    plt.xticks(rotation=15)
    
    # Ajustar o layout para não cortar os rótulos do eixo x
    plt.tight_layout()
    
    print("\nExibindo o gráfico...")
    plt.show()

if __name__ == '__main__':
    main()
