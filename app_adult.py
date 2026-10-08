import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Adult Census Income EDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Análise Exploratória - Base Adult (Census Income)")

# 1. & 2. Carregar o arquivo CSV e definir as colunas
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
    'marital-status', 'occupation', 'relationship', 'race', 'sex', 
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

@st.cache_data
def carregar_dados(url, columns):
    # Lendo o CSV. Usamos skipinitialspace=True porque o dataset possui espaços após as vírgulas
    return pd.read_csv(url, names=columns, skipinitialspace=True)

try:
    with st.spinner("Carregando base de dados do repositório UCI..."):
        df = carregar_dados(URL, COLUMNS)
except Exception as e:
    st.error(f"Erro ao carregar os dados da URL oficial: {e}")
    st.stop()

# Filtros na Sidebar
st.sidebar.header("Filtros")

# Filtro por Sexo
opcoes_sexo = sorted(df['sex'].dropna().unique().tolist())
sexo_selecionado = st.sidebar.multiselect(
    "Filtrar por Gênero:",
    options=opcoes_sexo,
    default=opcoes_sexo
)

# Filtro por Raça
opcoes_raca = sorted(df['race'].dropna().unique().tolist())
raca_selecionada = st.sidebar.multiselect(
    "Filtrar por Raça:",
    options=opcoes_raca,
    default=opcoes_raca
)

# Aplicar filtros
df_filtrado = df[
    (df['sex'].isin(sexo_selecionado)) & 
    (df['race'].isin(raca_selecionada))
]

if df_filtrado.empty:
    st.warning("Nenhum dado encontrado para a combinação de filtros selecionada.")
    st.stop()

# 3. Exibir o tamanho do DataFrame e as 5 primeiras linhas
st.subheader("Dimensões e Primeiras Linhas")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="Total de Linhas", value=f"{df_filtrado.shape[0]:,}")
with col2:
    st.metric(label="Total de Colunas", value=df_filtrado.shape[1])

st.write("**Primeiras 5 linhas:**")
st.dataframe(df_filtrado.head(), use_container_width=True)

# 4. Calcular a distribuição de renda separada por gênero
st.subheader("Distribuição de Renda por Gênero")
distribuicao = pd.crosstab(df_filtrado['sex'], df_filtrado['income'])
st.dataframe(distribuicao, use_container_width=True)

# 5. Plotar um gráfico de barras mostrando a contagem por raça, colorido por renda
st.subheader("Distribuição de Renda por Raça")
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(data=df_filtrado, x='race', hue='income', palette='Set2', ax=ax)
ax.set_title('Distribuição de Renda por Raça')
ax.set_xlabel('Raça')
ax.set_ylabel('Contagem')
ax.tick_params(axis='x', rotation=15)
fig.tight_layout()

# Exibir gráfico no Streamlit
st.pyplot(fig)
