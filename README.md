<br>

**\[[🇧🇷 Português](README.pt_BR.md)\] \[**[🇺🇸 English](README.md)**\]**


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




## 📚 Table of Contents (with GitHub Anchors)

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

Typical code steps already [here](https://github.com/Quantum-Software-Development/15-DataMining_Project_3_-Clustering_Comparison_KMeans_MeanShift_DBSCAN/blob/91ce4685c925253a2d054c9f89ebe16f00d27050/code/Project_3__Clustering_Comparison_KMeans_MeanShift_DBSCAN.ipynb) in the repo):

<br><br>

## 5.1 [Environment & load data]()

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

## 5.2 [Initial inspection & cleaning]()

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

## 5.3 [Scale numeric features & initial scatter plot]()

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


## 5.4 [K-distance graph (determine DBSCAN eps)]()

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

### [***PLOT 2***]() generated by `plt.plot(distances)` + `plt.show()`.

This plot is crucial for choosing the `eps` value — look for the **“elbow”** (sharp bend).

<br>




<br><br>





































<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>
<br><br>


## 21-  [Our Crew:]()


- 👨🏽‍🚀 **Andson Ribeiro** - [Slide into my inbox]()

- 👩🏻‍🚀 **Fabiana ⚡️ Campanari** - [Shoot me an email](mailto:fabicampanari@proton.me)

- 👨🏽‍🚀  **José Augusto de Souza Oliveira**  - [email]()

- 🧑🏼‍🚀 **Luan Fabiano**  - [email]()

- 👨🏽‍🚀 **Pedro Barrenco**  - [email]()
  
- 🧑🏼‍🚀 **Pedro Vyctor** - [Hit me up by email](mailto:pedro.vyctor00@gmail.com)



<br><br>


<!-- ========================== [Bibliographr ====================  -->

<br><br>


## [Bibliography]()


[1](). **Castro, L. N. & Ferrari, D. G.** (2016). *Introdução à mineração de dados: conceitos básicos, algoritmos e aplicações*. Saraiva.

[2](). **Ferreira, A. C. P. L. et al.** (2024). *Inteligência Artificial - Uma Abordagem de Aprendizado de Máquina*. 2nd Ed. LTC.

[3](). **Larson & Farber** (2015). *Estatística Aplicada*. Pearson.


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














