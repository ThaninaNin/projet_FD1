import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


# elbow pour K_means


def elbow(file_path):

  data = pd.read_csv(file_path, delimiter=';')
  # PREPROCESSING  AND NORMALISATION DES DONNées
  le = LabelEncoder()
  for col in data.columns:
    if data[col].dtype == 'object':
      data[col] = le.fit_transform(data[col].astype(str))
  imp = SimpleImputer(strategy='mean')
  data = pd.DataFrame(imp.fit_trans)

  scaler = MinMaxScaler()
  data1 = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=np.number)), columns=data.select_dtypes(include=np.number).columns)



  

  # Calcul de la variance expliquée pour différents nombres de classes
  variances = []
  for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data1)
    variances.append(kmeans.inertia_)

  # Tracé de la courbe du coude
  plt.plot(range(1, 11), variances)
  plt.xlabel('Nombre de classes')
  plt.ylabel('Variance expliquée')
  plt.title('Courbe du coude pour K-means')
  plt.show()

  st.write(plt)


  # Détermination du nombre optimal de classes
  diff_variances = np.diff(variances)
  diff2_variances = np.diff(diff_variances)
  n_clusters = np.argmin(diff2_variances) + 2
  st.write(f"Le nombre optimal de classes est {n_clusters}")






st.sidebar.title("Paramétres")


st.title("Projet Fouille de Données")
uploaded_file=st.file_uploader("Veuillez chosir votre dataset")
if uploaded_file is not None:
  elbow(uploaded_file)
