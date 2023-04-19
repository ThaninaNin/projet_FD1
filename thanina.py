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
import random
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OneHotEncoder
import time
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN




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



def kMeanCustom(k,data,data1):
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

  
def elbowKmeans(data1):
  variances = []
  for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(data1)
    variances.append(kmeans.inertia_)

 
  # Détermination du nombre optimal de classes
  diff_variances = np.diff(variances)
  diff2_variances = np.diff(diff_variances)
  n_clusters = np.argmin(diff2_variances) + 2

  return n_clusters,variances



def kmedoids(X, k, tmax=100):
    m, n = X.shape

    # Initialisation aléatoire des médoides
    medoids = np.array(random.sample(list(X), k))
    old_medoids = np.zeros((k, n))

    for i in range(tmax):
        # Étape d'affectation : attribution de chaque point au médoid le plus proche
        distances = cdist(X, medoids, metric='euclidean')
        labels = np.argmin(distances, axis=1)

        for i in range(k):
            indices = np.where(labels == i)
            cluster_points = X[indices]
            old_medoids[i, :] = medoids[i]
            medoids[i, :] = cluster_points[np.argmin(cdist(cluster_points, cluster_points, metric='euclidean').sum(axis=1)), :]

        if np.all(old_medoids == medoids):
            break

    # Calcul de la somme des distances entre chaque point et son médoid
    distances = cdist(X, medoids, metric='euclidean')
    labels = np.argmin(distances, axis=1)
    total_distance = sum(distances[range(len(labels)), labels])

    return medoids, labels, total_distance


def encodage(data):
  # Encodage des variables catégorielles avec one-hot encoding
  categorical_cols = [col for col in data.columns if data[col].dtype == 'object']
  for col in categorical_cols:
      onehot = pd.get_dummies(data[col], prefix=col)
      data = pd.concat([data, onehot], axis=1)
      data.drop(col, axis=1, inplace=True)

  # Suppression des lignes contenant des valeurs manquantes
  data.dropna(inplace=True)

  # Conversion du dataframe en array numpy
  X = data.values

  return X

def cleaningData(data):
  # Sélection des colonnes numériques
  numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns

  # Remplacement des valeurs manquantes par la moyenne
  data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

  # Encodage One-Hot des colonnes non numériques
  non_numeric_cols = data.select_dtypes(include=['object']).columns
  data = pd.get_dummies(data, columns=non_numeric_cols)

  # Normalisation des données
  scaler = StandardScaler()
  data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

  return data



st.title("Projet Fouille de Données")
uploaded_file = st.file_uploader("Veuillez chosir votre dataset")

if uploaded_file is not None :
  predata,postdata = preprocessing(uploaded_file)





if uploaded_file is not None:
  
  st.sidebar.title("Paramétres")
  methodes = st.sidebar.selectbox(
    'Quelle méthode voulez vous utilisez',
    ("preprocessing","Suprevisée","Non-Supervisée"))


  if "preprocessing" in methodes :


    
    progress_text = "Attendez le préprocessing"
    my_bar = st.progress(0, text=progress_text)

    for percent_complete in range(100):
      time.sleep(0.00001)
      my_bar.progress(percent_complete + 1, text=progress_text)

    st.success("Préprocessing avec succée", icon="✅")

    st.subheader('Data avant le préprocessing : ')
    st.write(predata)
    st.subheader('Data aprés le préprocessing :')
    st.write(postdata)

  if  "Non-Supervisée" in methodes :
    methode = st.sidebar.selectbox(
    'Quelle approche voulez vous utilisez',
    ("elbow for K-Means","K-Means","K-Medoid","Diana","Agnes","Dbscan"))



    if "elbow for K-Means" in methode :
      n_clusters,variances = elbowKmeans(postdata)
      chart_data = pd.DataFrame(range(1, 11),variances)

      st.line_chart(chart_data)
      st.success(f"Le nombre optimal de classes est {n_clusters}")

    
    
    elif  "K-Means" in methode :
      k = st.sidebar.slider('Choisir le nombre de cluster', 1, 30, 1)
      items,intraclasse,interclasse = kMeanCustom(k,predata,postdata)
      for label, indices in items.items():
        st.write("Cluster : ",label,predata.iloc[indices])
      st.sidebar.success(f"Intraclasse : {intraclasse}", icon="✅")
      st.sidebar.success(f"Interclasse : {interclasse}", icon="✅")

    elif "K-Medoid" in methode : 
      k = st.sidebar.slider('Choisir le nombre de cluster', 1, 30, 1)
      X = encodage(predata)
      medoids, labels, total_distance = kmedoids(X,k)
      # Calcul de l'intraclasse
      intraclasse = sum([cdist(X[labels == i], [medoids[i]], metric='euclidean').sum() for i in range(k)])

      # Calcul de l'interclasse
      interclasse = sum([len(X[labels == i]) * cdist([medoids[i]], [medoids.mean(axis=0)], metric='euclidean').sum() for i in range(k)])

      # Affichage des résultats
      st.subheader('Médoides finaux :')
      st.write(medoids)
      st.write('Somme des distances :', total_distance)
      st.write('Labels :', labels)
      st.sidebar.success(f"Intraclasse : {intraclasse}", icon="✅")
      st.sidebar.success(f"Interclasse : {interclasse}", icon="✅")
    
    elif "Dbscan" in methode :
      eps = st.sidebar.slider('Choisir epsilon', 0.1, 5.0, 0.1)
      min_samples = st.sidebar.slider('Choisir le nombre d’échantillons minimum ', 1, 20, 1)
      X = cleaningData(predata)
      dbscan = DBSCAN(eps=0.9, min_samples=5)
      dbscan.fit(X)
      # Affichage des clusters
      st.write(dbscan.labels_)
      # nombre de clusters (excluant le bruit)
      n_clusters = len(np.unique(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)
      st.write(f"Nombre de clusters: {n_clusters}")
      # Inertie intra-classe et inter-classe
      groups = {}
      for label in np.unique(dbscan.labels_):
          if label != -1:
              groups[label] = X[dbscan.labels_ == label]
              
      inertia_inter = 0
      inertia_intra = 0
      for label in groups:
          group = groups[label]
          group_mean = group.mean(axis=0)
          group_size = len(group)
          group_inertia = ((group - group_mean) ** 2).sum().sum()
          inertia_intra += group_inertia
          inertia_inter += group_size * ((group_mean - X.mean(axis=0)) ** 2).sum()

      st.sidebar.success(f"Inertie intra-classe : {inertia_intra}", icon="✅")
      st.sidebar.success(f"Inertie inter-classe : {inertia_inter}", icon="✅")



      



      



      

  
