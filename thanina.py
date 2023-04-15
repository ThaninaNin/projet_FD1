import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
import time




st.set_page_config(
    page_title="Projet Fouille de Données",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)

def preprocessing(file_path):
  data = pd.read_csv(file_path, delimiter=';')
  # Preprocess data
  le = LabelEncoder()
  for col in data.columns:
      if data[col].dtype == 'object':
          data[col] = le.fit_transform(data[col].astype(str))
  imp = SimpleImputer(strategy='mean')
  data = pd.DataFrame(imp.fit_transform(data), columns=data.columns)
  scaler = MinMaxScaler()
  data1 = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=np.number)), columns=data.select_dtypes(include=np.number).columns)
  return data,data1



def kMean(k,data,data1):
  # Cluster data
  kmeans = KMeans(n_clusters=k, n_init=10)
  kmeans.fit(data1)

  # Dictionnaire pour stocker les indices de chaque cluster
  clusters = {}
  for i, label in enumerate(kmeans.labels_):
    if label not in clusters:
      clusters[label] = [i]
    else:
      clusters[label].append(i)

  

  # Calcul de l'intraclasse
  intraclasse = kmeans.inertia_

  # Calcul de l'interclasse
  interclasse = pairwise_distances(kmeans.cluster_centers_, Y=None, metric='euclidean').sum()

  return clusters,intraclasse,interclasse

  







st.title("Projet Fouille de Données")
uploaded_file=st.file_uploader("Veuillez chosir votre dataset")
if uploaded_file is not None:
  
  
  
  

  
  data,data1 = preprocessing(uploaded_file)


  progress_text = "Attendez le préprocessing"
  my_bar = st.progress(0, text=progress_text)

  for percent_complete in range(100):
    time.sleep(0.00001)
    my_bar.progress(percent_complete + 1, text=progress_text)

  st.success("Préprocessing avec succée", icon="✅")

  st.sidebar.title("Paramétres")
  methodes = st.sidebar.multiselect('Quelle méthode voulez vous utilisez',["Suprevisée","Non-Supervisée"])

  if  "Non-Supervisée" in methodes :
    methode = st.sidebar.multiselect('Quelle approche voulez vous utilisez',["K-Means","K-Medoid","Diana","Agnes","Dbscan"])
    if  "K-Means" in methode :

      k = st.sidebar.slider('Choisir le nombre de cluster', 1, 30, 1)
      
      cluster = ["Tous"]
      for i in range(1,k+1):
        label = "Cluster : "
        labels = f"{label}{i}"
        cluster.append(labels)


      clusters = st.sidebar.multiselect('Quel Cluster voullez vous afficher',cluster)
      items,intraclasse,interclasse = kMean(k,data,data1)

      if "Tous" in clusters :
        # Affichage des éléments de chaque cluster
        for label, indices in items.items():
          st.write("Cluster : ",label,data.iloc[indices])
        st.success(f"Intraclasse : {intraclasse}", icon="✅")
        st.success(f"Interclasse : {interclasse}", icon="✅")


      

  
