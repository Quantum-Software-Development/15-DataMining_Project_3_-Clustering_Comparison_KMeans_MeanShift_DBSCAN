<br><br>

\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]


<br><br>


# 15- [Data Mining]()  / [Project 3 – Clustering Algorithms Exploration and Comparison]() - K-Means - Mean-Shift - Dbscan




<!-- ======================================= Start DEFAULT HEADER ===========================================  -->

<br><br>


[**Institution:**]() Pontifical Catholic University of São Paulo (PUC-SP)  
[**School:**]() Faculty of Interdisciplinary Studies  
[**Program:**]() Humanistic AI and Data Science
[**Semester:**]() 2nd Semester 2025  
Professor:  [***Professor Doctor in Mathematics Daniel Rodrigues da Silva***](https://www.linkedin.com/in/daniel-rodrigues-048654a5/)

<br><br>

#### <p align="center"> [![Sponsor Quantum Software Development](https://img.shields.io/badge/Sponsor-Quantum%20Software%20Development-brightgreen?logo=GitHub)](https://github.com/sponsors/Quantum-Software-Development)


<br><br>

<!--Confidentiality statement -->

#

<br><br><br>

> [!IMPORTANT]
> 
> ⚠️ Heads Up
>
> * Projects and deliverables may be made [publicly available]() whenever possible.
> * The course emphasizes [**practical, hands-on experience**]() with real datasets to simulate professional consulting scenarios in the fields of **Data Analysis and Data Mining** for partner organizations and institutions affiliated with the university.
> * All activities comply with the [**academic and ethical guidelines of PUC-SP**]().
> * Any content not authorized for public disclosure will remain [**confidential**]() and securely stored in [private repositories]().  
>


<br><br>

#

<!--END-->




<br><br><br><br>



<!-- PUC HEADER GIF
<p align="center">
  <img src="https://github.com/user-attachments/assets/0d6324da-9468-455e-b8d1-2cce8bb63b06" />
-->


<!-- video presentation -->


##### 🎶 Prelude Suite no.1 (J. S. Bach) - [Sound Design Remix]()

https://github.com/user-attachments/assets/4ccd316b-74a1-4bae-9bc7-1c705be80498

####  📺 For better resolution, watch the video on [YouTube.](https://youtu.be/_ytC6S4oDbM)


<br><br>


> [!TIP]
> 
>  This repository is a review of the Statistics course from the undergraduate program Humanities, AI and Data Science at PUC-SP.
>
> ### ☞ **Access Data Mining [Main Repository](https://github.com/Quantum-Software-Development/1-Main_DataMining_Repository)**
>
>


<!-- =======================================END DEFAULT HEADER ===========================================  -->


<br><br><br>


## Sumário

1. [Visão Geral do Projeto](#project-overview)
2. [O que tem neste repositório](#whats-in-this-repo)
3. [Início Rápido (executar o código)](#quick-start-run-the-code)
4. [Explicação Passo a Passo](#step-by-step-explanation-kid-friendly)
5. [Código Passo a Passo](#data-cleaning--preprocessing)
6. [Algoritmos usados (K-Means, Mean-Shift, DBSCAN)](#algorithms-used-k-means-mean-shift-dbscan)
7. [Como escolhemos o eps do DBSCAN (gráfico K-distância)](#how-we-chose-dbscan-eps-k-distance-graph)
8. [Visualização](#visualization--style-dark--turquoise)
9. [Resumo dos resultados & interpretação](#results-summary--interpretation)
10. [Próximos passos & sugestões](#next-steps--suggestions)
    11 [Requisitos & ambiente](#requirements--environment)
11. [Referências](#references)
12. [Licença & créditos](#license--credits)

<br><br>



## 1. [Visão Geral do Projeto]()

Este projeto carrega um dataset CSV (`Dados-Grupo4.csv`), inspeciona e limpa os dados, aplica normalização (feature scaling) e compara três algoritmos de clusterização: **K-Means, Mean-Shift e DBSCAN**. Inclui gráficos em estilo turquesa escuro e explicações claras para ajudar qualquer pessoa a entender o fluxo de trabalho e os resultados.


<br><br>

## 2. [O que tem neste Repositório]()

* `Dados-Grupo4.csv` — arquivo principal do dataset.
* `notebook.ipynb` ou `run_clustering.py` — código principal responsável por carregar, limpar, clusterizar e plotar.
* `README.md` — esta documentação.
* `requirements.txt` — lista de pacotes Python necessários.

<br><br>


## 3. [Início Rápido (executar o código)]()

[3.1]()- Abra o Google Colab ou seu ambiente Python local.

<br>

[3.2]()- Faça o upload de `Dados-Grupo4.csv` para a pasta de trabalho.

<br>

[3.3]()- Instale as dependências:

<br>

```bash
pip install -r requirements.txt
```

<br>

[3.4]()- Execute o `notebook.ipynb` célula por célula ou rode:

<br>

```bash
python run_clustering.py
```

<br>


[3.5]()- **Exemplo de `requirements.txt`:**

<br>

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

<br><br>

## 4. [Explicação passo a passo]()

* Abrimos a tabela (CSV) — como abrir uma planilha.
* Contamos quantas linhas (linhas) e colunas (tipos de informação) ela tem.
* Observamos números básicos: médias, menores, maiores — ajuda a entender os dados.
* Removemos qualquer coluna extra chamada "Unnamed: 0", se existir.
* Se algumas células estiverem vazias, preenchemos com o valor mais comum (moda).
* Se duas linhas forem idênticas, excluímos as duplicatas.
* Escalonamos os números para que valores grandes não dominem os padrões.
* Usamos três métodos para agrupar pontos (K-Means, Mean-Shift, DBSCAN).
* Desenhamos os grupos como imagens com fundo escuro e cor turquesa.
* Comparamos os resultados e explicamos o que cada método descobriu.

<br><br>

## 5. [Código Passo a Passo]()

Etapas típicas do código já estão **[aqui](https://github.com/Quantum-Software-Development/15-DataMining_Project_3_-Clustering_Comparison_KMeans_MeanShift_DBSCAN/blob/91ce4685c925253a2d054c9f89ebe16f00d27050/code/Project_3__Clustering_Comparison_KMeans_MeanShift_DBSCAN.ipynb)** no repositório):

<br><br>

## 5.1 - [Ambiente & carregamento dos dados]()

[***O que faz***](): importa bibliotecas, define tema escuro e paleta turquesa, carrega o CSV e imprime o formato (shape).

<br>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configura matplotlib para fundo escuro
plt.style.use('dark_background')
sns.set_palette('GnBu_r')

# Carrega o dataset
df = pd.read_csv('/content/Dados-Grupo4.csv')

# Exibe o número de linhas e colunas
print(f"Dataset possui {df.shape[0]} linhas e {df.shape[1]} colunas.")
```

<br><br>


## 5.2 - [Inspeção inicial & limpeza]()

[***O que faz***](): `df.describe()`, remove `'Unnamed: 0'` se existir, preenche valores faltantes com a moda, remove duplicatas.

<br>

```python
print(df.describe())

# Remove coluna extra se existir
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Preenche valores faltantes
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# Remove duplicatas
df = df.drop_duplicates()
```

<br><br>


## 5.3 - [Escalonar features numéricas & scatter plot inicial]()

[***O que faz***](): padroniza as variáveis numéricas e gera o gráfico de dispersão inicial (tamanho 12×8).

<br>

```python
from sklearn.preprocessing import StandardScaler

columns_to_scale = ['Coluna1', 'Coluna2']  # adapte se suas colunas forem diferentes
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]), columns=columns_to_scale)

# --- PLOT 1: Scatter plot inicial ---
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df_scaled['Coluna1'], y=df_scaled['Coluna2'])
plt.title('Gráfico de Dispersão Inicial dos Dados Escalonados')
plt.xlabel('Coluna1 Escalonada')
plt.ylabel('Coluna2 Escalonada')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

<br>

### [***PLOT 1***]() — Scatter Inicial

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f2d7894-6837-4f42-b689-2675e6e78cab" width="100%">
</p>


<br><br>

> [!TIP]
> 
> 👌🏻
> 
> [***To save***](): add plt.savefig('initial_scatter.png', dpi=300, bbox_inches='tight') before de plt.show()
> 

<br><br>


















































<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>


<!-- ======================================= Start Footer ===========================================  -->


<br><br>


## 💌 [Let the data flow... Ping Me !](mailto:fabicampanari@proton.me)

<br><br>



#### <p align="center">  🛸๋ My Contacts [Hub](https://linktr.ee/fabianacampanari)


<br>

### <p align="center"> <img src="https://github.com/user-attachments/assets/517fc573-7607-4c5d-82a7-38383cc0537d" />




<br><br><br>

<p align="center">  ────────────── 🔭⋆ ──────────────


<p align="center"> ➣➢➤ <a href="#top">Back to Top </a>

<!--
<p align="center">  ────────────── ✦ ──────────────
-->



<!-- Programmers and artists are the only professionals whose hobby is their profession."

" I love people who are committed to transforming the world "

" I'm big fan of those who are making waves in the world! "

##### <p align="center">( Rafael Lain ) </p>   -->

#

###### <p align="center"> Copyright 2025 Quantum Software Development. Code released under the [MIT License license.](https://github.com/Quantum-Software-Development/Math/blob/3bf8270ca09d3848f2bf22f9ac89368e52a2fb66/LICENSE)

















