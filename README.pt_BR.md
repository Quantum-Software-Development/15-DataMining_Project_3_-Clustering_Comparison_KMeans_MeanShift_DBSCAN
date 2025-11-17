<br><br>

\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇬🇧 English](README.md)**\]


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
> [***Para Salvar***](): add plt.savefig('initial_scatter.png', dpi=300, bbox_inches='tight') before de plt.show()
> 

<br><br>


## 5.4 - [Gráfico K-distance (determinar eps do DBSCAN)]()

[***O que faz***](): calcula a distância para o 4º vizinho mais próximo de cada ponto e plota as distâncias ordenadas — o gráfico K-distance usado para escolher o valor de *eps*.

<br>

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

neigh = NearestNeighbors(n_neighbors=4)
neigh.fit(df_scaled)
distances, indices = neigh.kneighbors(df_scaled)
distances = np.sort(distances[:, 3], axis=0)  # distância até o 4º vizinho

# --- PLOT 2: Gráfico K-distance ---
plt.figure(figsize=(12, 8))
plt.plot(distances)
plt.title('Gráfico K-distance para DBSCAN')
plt.xlabel('Pontos ordenados pela distância')
plt.ylabel('Epsilon (Distância)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```

<br>


### [***PLOT 2***]() - gerado por `plt.plot(distances)` + `plt.show()`.

Este gráfico é essencial para escolher o valor de `eps` — procure o **“cotovelo”** (a curva onde há uma mudança brusca).

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b680cc5c-8b24-4514-9bb0-5df6d56243da" width="100%">
</p>

<br><br>


> [!TIP]
> 
> 👌🏻
> 
> [***Para Salvar***](): plt.savefig('k_distance.png', dpi=300, bbox_inches='tight').
>


<br><br>


## 5.5 - [Aplicar algoritmos de clusterização & visualização combinada]()

[***O que faz***](): executa K-Means, Mean-Shift e DBSCAN; armazena os rótulos (labels); plota os três resultados lado a lado.

<br>

```python
from sklearn.cluster import KMeans, MeanShift, DBSCAN
from sklearn.cluster import estimate_bandwidth

# K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df_scaled['kmeans_labels'] = kmeans.fit_predict(df_scaled[['Coluna1','Coluna2']])

# Mean-Shift
bandwidth = estimate_bandwidth(df_scaled[['Coluna1', 'Coluna2']], quantile=0.2, n_samples=len(df_scaled))
meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True)
df_scaled['meanshift_labels'] = meanshift.fit_predict(df_scaled[['Coluna1','Coluna2']])

# DBSCAN (escolha eps pelo gráfico K-distance)
dbscan = DBSCAN(eps=0.25, min_samples=4)
df_scaled['dbscan_labels'] = dbscan.fit_predict(df_scaled[['Coluna1','Coluna2']])

# --- PLOT 3: Comparação (três subplots) ---
plt.figure(figsize=(20, 7))

plt.subplot(1, 3, 1)
sns.scatterplot(x=df_scaled['Coluna1'], y=df_scaled['Coluna2'], hue=df_scaled['kmeans_labels'], palette='GnBu_r', legend='full')
plt.title('Clusterização K-Means (K=3)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 2)
sns.scatterplot(x=df_scaled['Coluna1'], y=df_scaled['Coluna2'], hue=df_scaled['meanshift_labels'], palette='GnBu_r', legend='full')
plt.title(f'Mean-Shift (bandwidth={bandwidth:.2f})')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
sns.scatterplot(x=df_scaled['Coluna1'], y=df_scaled['Coluna2'], hue=df_scaled['dbscan_labels'], palette='GnBu_r', legend='full')
plt.title('DBSCAN (eps=0.25, min_samples=4)')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
```

<br>

### [***PLOT 3***]() - Comparação:** a sequência `plt.subplot(...); sns.scatterplot(...); plt.show()` gera os três gráficos juntos (K-Means, Mean-Shift, DBSCAN).

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/79a7f744-7acc-46cd-9cb5-e7532051bb4d" width="100%">
</p>

<br><br>

> [!TIP]
> 
> 👌🏻
> 
> [***Para Salvar***](): each subplot as a single image: before `plt.show()`, use `plt.savefig('comparison_three_algorithms.png', dpi=300, bbox_inches='tight')`.
>
> [***Para Salvar***](): separate images for each algorithm, move each subplot block into separate cells and save them individually.
>

<br><br>


## 5.6 - [Exibir contagem de clusters & métricas opcionais]()

[***O que faz***](): mostra quantos clusters cada método encontrou e, opcionalmente, calcula o *silhouette score*.

<br>

```python
print(f"Número de clusters K-Means: {df_scaled['kmeans_labels'].nunique()}")
print(f"Número de clusters Mean-Shift: {df_scaled['meanshift_labels'].nunique()}")
print(f"Número de clusters DBSCAN (excluindo ruído -1): {df_scaled['dbscan_labels'].nunique() - (1 if -1 in df_scaled['dbscan_labels'].unique() else 0)}")

# Opcional: silhouette
from sklearn.metrics import silhouette_score
print('Silhouette K-Means:', silhouette_score(df_scaled[['Coluna1','Coluna2']], df_scaled['kmeans_labels']))
```

<br><br>


<br><br>

## 6. [Resumo dos resultados & interpretação]()

* [**K-Means (K=3):**]() Encontrou 3 clusters — linha de base padrão, assume grupos arredondados.
* [**Mean-Shift:**]() Encontrou 4 clusters — se adapta automaticamente às regiões densas.
* [**DBSCAN**]() (eps escolhido pelo gráfico K-distance, min_samples=4): Encontrou 5 clusters + ruído — bom para grupos densos e detecção de outliers.
* [**Interpretação:**]() Cada método agrupa os pontos de forma diferente, dependendo das regras. É como organizar brinquedos por cor vs por proximidade na prateleira — os montes serão diferentes.

<br><br>


## 7. [Próximos passos & sugestões]()

[-]() Calcular métricas como *silhouette score* ou índice de Davies-Bouldin para comparar os métodos numericamente.

[-]() Testar outros valores de K para o K-Means; experimentar diferentes quantis para o *bandwidth* do Mean-Shift.

[-]() Com mais de 2 features, usar PCA/t-SNE/UMAP para visualização.

[-]() Se o DBSCAN encontrar muito ruído, ajustar *eps*/*min_samples* ou testar HDBSCAN.

[-]() Criar um slide comparando os três gráficos lado a lado, com conclusões em uma frase para cada.

[-]() Exemplo de código para *silhouette*:


<br>

```python
from sklearn.metrics import silhouette_score
print('KMeans silhouette:', silhouette_score(df_scaled, kmeans_labels))
```


<br><br>

## 8. [Requisitos & ambiente]()

[-]() Python 3.8 ou superior

[-]() pandas, numpy, matplotlib, seaborn, scikit-learn

[-]() Opcional: Jupyter Notebook ou Google Colab

<br><br>


<br><br>


## 9.  [Our Crew:]()


- 👨🏽‍🚀 **Andson Ribeiro** - [Slide into my inbox]()

- 👩🏻‍🚀 **Fabiana ⚡️ Campanari** - [Shoot me an email](mailto:fabicampanari@proton.me)

- 👨🏽‍🚀  **José Augusto de Souza Oliveira**  - [email]()

- 🧑🏼‍🚀 **Luan Fabiano**  - [email]()

- 👨🏽‍🚀 **Pedro Barrenco**  - [email]()
  
- 🧑🏼‍🚀 **Pedro Vyctor** - [Hit me up by email](mailto:pedro.vyctor00@gmail.com)



<br><br>


<!-- ========================== [Bibliographry ====================  -->

<br><br>



<br><br>

## 10. [Bibliografia]()

[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ester, M., Kriegel, H.-P., Sander, J., & Xu, X.** (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*.

[3](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial — Uma Abordagem de Aprendizado de Máquina*. 2ª Ed. LTC.

[4](). **Larson, R. & Farber, B.** (2015). *Estatística Aplicada*. Pearson.

[5](). **MacQueen, J.** (1967). *Some Methods for Classification and Analysis of Multivariate Observations* — origem do K-Means.

[6](). **Meer, P. & Comaniciu, D.** (2002). *Mean Shift: A Robust Approach Toward Feature Space Analysis*.

[7](). **Documentação do scikit-learn** — algoritmos de clusterização.

[8](). **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** — referência geral de Data Science.



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

















