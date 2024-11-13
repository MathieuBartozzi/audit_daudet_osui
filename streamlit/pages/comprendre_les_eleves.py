
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os


st.set_page_config(layout="wide")

# Chemin relatif pour accéder au dossier 'data_app' depuis 'pages'
directory_path = "data_app"

def load_all_csvs_in_directory(directory_path):
    # Dictionnaire pour stocker les DataFrames
    dataframes = {}

    # Vérifier si le dossier existe
    if not os.path.exists(directory_path):
        st.error("Le dossier spécifié n'existe pas.")
        return None

    # Lister et charger tous les fichiers CSV dans le dossier
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            dataframes[file_name] = pd.read_csv(file_path)

    # st.write("Tous les fichiers CSV ont été chargés avec succès.")
    return dataframes

# Utilisation dans l'application


st.title("Chargement de tous les fichiers CSV dans un dossier")
dataframes = load_all_csvs_in_directory(directory_path)

# st.write("Clés disponibles dans dataframes :", list(dataframes.keys()))


# Charger les données du fichier profil_counts.csv
profil_counts = dataframes['profil_counts.csv']

# Vérifier si 'profil_counts' est une série ou un DataFrame
if isinstance(profil_counts, pd.DataFrame):
    # Supposons que la première colonne soit les index et la seconde les valeurs
    profil_counts = profil_counts.iloc[:, 0]

# Création du graphique en camembert avec Plotly
fig = px.pie(
    names=[f"Profil {i + 1}" for i in profil_counts.index],  # Remplacement de "Cluster" par "Profil" et commencement à 1
    values=profil_counts.values,  # Utilisation des valeurs de la série ou de la colonne sélectionnée
    hole=0.3,
    color_discrete_sequence=px.colors.qualitative.G10
    )


st.subheader('Répartition des élèves par profil')
col1, col2 =st.columns([1,2])

with col1:
    with st.popover("Profil 1"):
        st.markdown("""
        Ce groupe représente des **élèves avec des difficultés scolaires importantes**, ayant une moyenne annuelle de **11,50** et un taux d'absentéisme très élevé (**50,47**). Ils ont un nombre modéré de retards (**12,35**) et peu de punitions (**2,86**), ainsi que des passages à l'infirmerie faibles (**2,10**). Ces élèves sont majoritairement des filles (environ 73 %). Ce profil pourrait indiquer des élèves faisant face à des situations personnelles ou familiales particulières.
        """)

    with st.popover("Profil 2"):
        st.markdown("""
        Les élèves de ce profil ont des **résultats modérément bons** avec une moyenne annuelle de **14,30** mais présentent un nombre très élevé de retards (**23,04**) et de punitions (**15,40**). Le nombre d'absences est modéré (**14,82**) et les passages à l'infirmerie sont également modérés (**5,30**). Ce profil est majoritairement composé de garçons (environ 68 %), ce qui suggère des **difficultés comportementales** et un besoin de soutien pour améliorer leur discipline et assiduité.
        """)

    with st.popover("Profil 3"):
        st.markdown("""
        Ce profil regroupe des **élèves performants scolairement** avec une moyenne annuelle élevée de **15,68** et peu d'absences (**11,95**), de retards (**5,24**), ou de punitions (**1,03**). Les passages à l'infirmerie sont également faibles (**2,93**). Ce groupe est exclusivement composé de filles. Ce profil semble stable et autonome.
        """)

    with st.popover("Profil 4"):
        st.markdown("""
        Les élèves de ce groupe présentent également des **résultats scolaires faibles** avec une moyenne annuelle de **12,43**, mais se démarquent par un nombre extrêmement élevé de passages à l'infirmerie (**108**). Leurs absences sont modérées (**19,25**) ainsi que leurs retards (**6,50**) et punitions (**5,25**). Ce groupe est principalement composé de filles (environ 75 %).
        """)

    with st.popover("Profil 5"):
        st.markdown("""
        Ce groupe représente des **garçons ayant de bonnes performances scolaires**, avec une moyenne annuelle de **14,54**. Ils ont peu d'absences (**13,75**), de retards (**4,74**), de punitions (**2,02**), et de passages à l'infirmerie (**2,39**). Ce profil semble stable et autonome.
        """)

with col2 :
    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig)










st.title("titre 1")
