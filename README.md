<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇬🇧 English](README.md)**\]**


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




##  Table of Contents 

1. [Project Overview](#project-overview)
2. [What's in this repo](#whats-in-this-repo)
3. [Quick Start (run the code)](#quick-start-run-the-code)
4. [Step-by-Step Explanation](#step-by-step-explanation-kid-friendly)
5. [Code Step-by-Step](#data-cleaning--preprocessing)
6. [Algorithms used (K-Means, Mean-Shift, DBSCAN)](#algorithms-used-k-means-mean-shift-dbscan)
7. [How we chose DBSCAN eps (K-distance graph)](#how-we-chose-dbscan-eps-k-distance-graph)
8. [Visualization](#visualization--style-dark--turquoise)
9. [Results summary & interpretation](#results-summary--interpretation)
10. [Next steps & suggestions](#next-steps--suggestions)
11 [Requirements & environment](#requirements--environment)
12. [References](#references)
13. [License & credits](#license--credits)



<br><br>


## 1. [Project Overview]()

This project loads a CSV dataset (`Dados-Grupo4.csv`), inspects and cleans it, applies feature scaling, and compares three clustering algorithms: **K-Means, Mean-Shift, and DBSCAN**. It includes dark turquoise plots and clear explanations to help anyone understand the workflow and results.


<br><br>

## 2. [What's in this Repo]()

- `Dados-Grupo4.csv` — main dataset file.
- `notebook.ipynb` or `run_clustering.py` — main code handling loading, cleaning, clustering, and plotting.
- `README.md` — this documentation.
- `requirements.txt` — list of Python packages needed.

<br><br>

## 3. [Quick Start (run the code)]()

[3.1]()- Open Colab or your local Python environment.

<br>

[3.2]()- Upload `Dados-Grupo4.csv` to the working folder.

<br>

[3.3]()- Install dependencies:

<br>

```bash
pip install -r requirements.txt
```

<br>

[3.4]()- Either run `notebook.ipynb` cell by cell or execute:

<br>

```bash
python run_clustering.py
```

<br>

[3.5]()- **Example `requirements.txt`:**

<br>

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

<br><br>

## 4. [Step-by-step explanation (kid-friendly)]()

- We open the table (CSV) — like opening a spreadsheet.
- Count how many rows (lines) and columns (types of information) it has.
- Look at basic numbers: averages, smallest, biggest — helps understand the data.
- Remove any extra "Unnamed: 0" column if present.
- If some boxes are empty, fill them with the most common value (mode).
- If two rows are identical, delete duplicates.
- Scale the numbers so large values don't dominate the patterns.
- Use three methods to group points (K-Means, Mean-Shift, DBSCAN).
- Draw the groups as pictures with a dark background and turquoise color.
- Compare the results and explain what each method discovered.



<br><br>



## 5. [Code Step-by-Step]()

* Typical code steps already [here](https://github.com/Quantum-Software-Development/15-DataMining_Project_3_-Clustering_Comparison_KMeans_MeanShift_DBSCAN/blob/91ce4685c925253a2d054c9f89ebe16f00d27050/code/Project_3__Clustering_Comparison_KMeans_MeanShift_DBSCAN.ipynb) in the repo):

<br>

* [Pedro Victor’s Implementation]()

<br><br>

## 5.1 - [Environment & load data]()

[***What it does***](): import libraries, set dark theme and turquoise palette, load CSV and print shape.

<br>

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib for dark background
plt.style.use('dark_background')
sns.set_palette('GnBu_r')

# Load the dataset
df = pd.read_csv('/content/Dados-Grupo4.csv')

# Display the number of rows and columns
print(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
```

<br><br>

## 5.2 - [Initial inspection & cleaning]()

[***What it does***](): df.describe(), remove 'Unnamed: 0' if exists, fill missing values with mode, drop duplicates. 

<br>

```python
print(df.describe())

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# fill missing
for col in df.columns:
    if df[col].isnull().any():
        df[col] = df[col].fillna(df[col].mode()[0])

# drop duplicates
df = df.drop_duplicates()
```

<br><br>

## 5.3 - [Scale numeric features & initial scatter plot]()

[***What it does***](): standardize numeric features and produce the initial scatter plot (figsize 12×8). 

<br>

```python
from sklearn.preprocessing import StandardScaler

columns_to_scale = ['Coluna1', 'Coluna2']  # adapt if columns differ
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[columns_to_scale]), columns=columns_to_scale)

# --- PLOT 1: Initial scatter plot ---
plt.figure(figsize=(12, 8))
sns.scatterplot(x=df_scaled['Coluna1'], y=df_scaled['Coluna2'])
plt.title('Initial Scatter Plot of Scaled Data')
plt.xlabel('Scaled Coluna1')
plt.ylabel('Scaled Coluna2')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```
<br>

### [***PLOT 1***]() — Initial Scatter

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


## 5.4 - [K-distance graph (determine DBSCAN eps)]()

[***What it does***](): computes distance to 4th nearest neighbor for each point and plots sorted distances — the K-distance graph used to pick eps.

<br>

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np

neigh = NearestNeighbors(n_neighbors=4)
neigh.fit(df_scaled)
distances, indices = neigh.kneighbors(df_scaled)
distances = np.sort(distances[:, 3], axis=0)  # distance to 4th NN

# --- PLOT 2: K-distance graph ---
plt.figure(figsize=(12, 8))
plt.plot(distances)
plt.title('K-distance Graph for DBSCAN')
plt.xlabel('Data Points sorted by Distance')
plt.ylabel('Epsilon (Distance)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```


<br>

### [***PLOT 2***]() - generated by `plt.plot(distances)` + `plt.show()`.

This plot is crucial for choosing the `eps` value — look for the **“elbow”** (sharp bend).

<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/b680cc5c-8b24-4514-9bb0-5df6d56243da" width="100%">
</p>

<br><br>

> [!TIP]
> 
> 👌🏻
> 
> [***To save***](): plt.savefig('k_distance.png', dpi=300, bbox_inches='tight').
>


<br><br>


## 5.5 - [Apply clustering algorithms & combined visualization]()

[***What it does***]():  runs K-Means, Mean-Shift and DBSCAN; stores labels; plots the three results side-by-side.


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

# DBSCAN (choose eps from k-distance)
dbscan = DBSCAN(eps=0.25, min_samples=4)
df_scaled['dbscan_labels'] = dbscan.fit_predict(df_scaled[['Coluna1','Coluna2']])

# --- PLOT 3: Comparison (three subplots) ---
plt.figure(figsize=(20, 7))

plt.subplot(1, 3, 1)
sns.scatterplot(x=df_scaled['Coluna1'], y=df_scaled['Coluna2'], hue=df_scaled['kmeans_labels'], palette='GnBu_r', legend='full')
plt.title('K-Means Clustering (K=3)')
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

### [***PLOT 3***]() - Comparison:** the sequence `plt.subplot(...); sns.scatterplot(...); plt.show()` generates the three plots together (K-Means, Mean-Shift, DBSCAN).


<br>

<p align="center">
  <img src="https://github.com/user-attachments/assets/79a7f744-7acc-46cd-9cb5-e7532051bb4d" width="100%">
</p>

<br><br>

> [!TIP]
> 
> 👌🏻
> 
> [***To save***](): each subplot as a single image: before `plt.show()`, use `plt.savefig('comparison_three_algorithms.png', dpi=300, bbox_inches='tight')`.
>
> [***To save***](): separate images for each algorithm, move each subplot block into separate cells and save them individually.
>

<br><br>



## 5.6 - [Print cluster counts & optional metrics]()

[***What it does***](): prints how many clusters each method found and optionally computes silhouette score.

<br>


```python
print(f"Number of K-Means clusters: {df_scaled['kmeans_labels'].nunique()}")
print(f"Number of Mean-Shift clusters: {df_scaled['meanshift_labels'].nunique()}")
print(f"Number of DBSCAN clusters (excluding noise -1): {df_scaled['dbscan_labels'].nunique() - (1 if -1 in df_scaled['dbscan_labels'].unique() else 0)}")

# Optional: silhouette
from sklearn.metrics import silhouette_score
print('KMeans silhouette:', silhouette_score(df_scaled[['Coluna1','Coluna2']], df_scaled['kmeans_labels']))
```


<br><br>


## 6. [Results summary \& interpretation]()

- [**K-Means (K=3):**]() Found 3 clusters — standard baseline, assumes round groups.
- [**Mean-Shift:**]() Found 4 clusters — adapts to dense regions automatically.
- [**DBSCAN**]() (eps chosen from K-distance, min_samples=4): Found 5 clusters + noise — good for dense groups and spotting outliers.
- [**Interpretation:**]() Each method groups points differently depending on the rules. Like sorting toys by color vs by how close they are on a shelf — the piles will be different.


<br><br>


## 7. [Next steps \& suggestions]()

[-]() Compute metrics like silhouette score or Davies-Bouldin index to compare methods numerically.

[-]() Try other K values for K-Means; test different quantiles for Mean-Shift bandwidth.

[-]() With more than 2 features, try PCA/t-SNE/UMAP for visualization.

[-]() If DBSCAN finds too much noise, adjust eps/min_samples or test HDBSCAN.

[-]() Make a slide comparing all three plots side by side, with one-sentence conclusions.

[-]() Example silhouette code:

<br>

```python
from sklearn.metrics import silhouette_score
print('KMeans silhouette:', silhouette_score(df_scaled, kmeans_labels))
```

<br><br>

## 8. [Requirements \& environment]()

[-]() Python 3.8 or higher

[-]() pandas, numpy, matplotlib, seaborn, scikit-learn
  
[-]() Optional: Jupyter Notebook or Google Colab


<br><br>


## 9.  [Our Crew:]()


- 👨🏽‍🚀 **Andson Ribeiro** - [Slide into my inbox]()

- 👩🏻‍🚀 **Fabiana ⚡️ Campanari** - [Shoot me an email](mailto:fabicampanari@proton.me)

- 👨🏽‍🚀  **José Augusto de Souza Oliveira**  - [email]()

- 🧑🏼‍🚀 **Luan Fabiano**  - [email]()

- 👨🏽‍🚀 **Pedro Barrenco**  - [email]()
  
- 🧑🏼‍🚀 **Pedro Vyctor** - [Hit me up by email](mailto:pedro.vyctor00@gmail.com)



<br><br>


<!-- ========================== [Bibliographr ====================  -->

<br><br>


## 10. [Bibliography]()


[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ester, M., Kriegel, H.-P., Sander, J., & Xu, X.** (1996). *A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise*.

[3](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial — Uma Abordagem de Aprendizado de Máquina*. 2nd Ed. LTC.

[4](). **Larson, R. & Farber, B.** (2015). *Estatística Aplicada*. Pearson.

[5](). **MacQueen, J.** (1967). *Some Methods for Classification and Analysis of Multivariate Observations* — origin of K-Means.

[6](). **Meer, P. & Comaniciu, D.** (2002). *Mean Shift: A Robust Approach Toward Feature Space Analysis*.

[7](). **scikit-learn documentation** — clustering algorithms.

[8](). **Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow** — general DS reference.



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














