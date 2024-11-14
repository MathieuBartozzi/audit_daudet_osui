
import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.subplots as sp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

##### CHARGEMENT DES DONNÉES ######

# Fonction de chargement des données avec cache
directory_path = 'data_app'

@st.cache_data
def load_data(directory_path='data_app'):
    """
    Charge les données depuis les fichiers CSV spécifiés dans le dossier donné.

    Parameters:
        directory_path (str): Chemin du dossier contenant les fichiers CSV.

    Returns:
        tuple: Contient tous les DataFrames chargés depuis les fichiers CSV.
    """
    # Liste des fichiers et leurs chemins complets
    df_eleve = pd.read_csv(os.path.join(directory_path, "Eleve.csv"), sep=';')
    df_notes_devoir = pd.read_csv(os.path.join(directory_path, "df_notes_devoir.csv"), sep=';')
    df_absenceseleves = pd.read_csv(os.path.join(directory_path, "Absenceseleves.csv"), sep=';')
    df_retards = pd.read_csv(os.path.join(directory_path, "Retards.csv"), sep=';')
    df_punition = pd.read_csv(os.path.join(directory_path, "Punition.csv"), sep=';', error_bad_lines=False, na_values=[''])
    df_passagesinfirmerie = pd.read_csv(os.path.join(directory_path, "Passagesinfirmerie.csv"), sep=';')
    df_sanction = pd.read_csv(os.path.join(directory_path, "Sanction.csv"), sep=';')

    return df_eleve, df_notes_devoir, df_absenceseleves, df_retards, df_punition, df_passagesinfirmerie, df_sanction

# Appel de la fonction pour charger les données
df_eleve, df_notes_devoir, df_absenceseleves, df_retards, df_punition, df_passagesinfirmerie, df_sanction = load_data(directory_path)


############ CREATION DU DATAFRAME ELEVES ############

# Listes des classes pour chaque niveau
lycee_classes = ['1A', '1B/C', '2A', '2B', '2C', 'Terminale']
college_classes = ['3A', '3B', '3C', '4A', '4B', '4C', '5A', '5B', '5C', '6A', '6B', '6C']

# Fonction pour déterminer le niveau
def determiner_niveau(classe):
    if classe in lycee_classes:
        return 'lycée'
    elif classe in college_classes:
        return 'collège'
    else:
        return 'N/A'  # Si la classe ne correspond à aucun niveau défini

# Application de la fonction pour créer la colonne 'niveau'
df_notes_devoir['niveau'] = df_notes_devoir['classes'].apply(determiner_niveau)

# Renommer la colonne `ident` dans df_eleve pour correspondre à `eleve_id`
df_eleve.rename(columns={'ident': 'eleve_id'}, inplace=True)

# Calcul de la note pondérée en normalisant sur 20 et appliquant le coefficient
df_notes_devoir['note_ponderee'] = (df_notes_devoir['note'] / df_notes_devoir['sur']) * 20 * df_notes_devoir['coeff']



tab1, tab2 = st.tabs(["**RESULTATS**", "**VIE SCOLAIRE**"])



############################## VIE SCOLAIRE  ##############################
with tab1 :

    ############ CREATION DU DATAFRAME ELEVES ############
    # Calcul de la moyenne annuelle pondérée pour chaque élève
    notes_annuelles = df_notes_devoir.groupby('eleve_id').apply(
        lambda x: x['note_ponderee'].sum() / x['coeff'].sum()
    ).reset_index(name='moyenne_annuelle')

    # Calcul des indicateurs d'absences, retards, punitions, passages à l'infirmerie
    absences_total = df_absenceseleves.groupby('eleve_id').size().reset_index(name='total_absences')
    retards_total = df_retards.groupby('eleve_id').size().reset_index(name='total_retards')
    punitions_total = df_punition.groupby('eleve_id').size().reset_index(name='total_punitions')
    infirmerie_total = df_passagesinfirmerie.groupby('eleve_id').size().reset_index(name='total_passages_infirmerie')

    # Fusionner tous les indicateurs dans un DataFrame unique avec `sexe`
    # indicateurs_eleves = notes_annuelles.merge(absences_total, on='eleve_id', how='left') \
    #                             .merge(retards_total, on='eleve_id', how='left') \
    #                             .merge(punitions_total, on='eleve_id', how='left') \
    #                             .merge(infirmerie_total, on='eleve_id', how='left') \
    #                             .merge(df_eleve[['eleve_id', 'sexe']], on='eleve_id', how='left')

    # Ajout de la colonne 'niveau' de df_notes_devoir à indicateurs_eleves
    indicateurs_eleves = notes_annuelles.merge(absences_total, on='eleve_id', how='left') \
                             .merge(retards_total, on='eleve_id', how='left') \
                             .merge(punitions_total, on='eleve_id', how='left') \
                             .merge(infirmerie_total, on='eleve_id', how='left') \
                             .merge(df_eleve[['eleve_id', 'sexe']], on='eleve_id', how='left') \
                             .merge(df_notes_devoir[['eleve_id', 'niveau']].drop_duplicates(), on='eleve_id', how='left')


    # Encoder la variable `sexe` : 0 pour "F" (filles) et 1 pour "M" (garçons)
    indicateurs_eleves['sexe'] = indicateurs_eleves['sexe'].map({'F': 0, 'M': 1})

    # Remplir les valeurs manquantes par 0
    indicateurs_eleves.fillna(0, inplace=True)
    ############ CREATION DU DATAFRAME ELEVES ✅ ############

    ############ PROFILAGE ############

    # Normalisation des indicateurs, incluant la variable `sexe`
    scaler = StandardScaler()
    indicateurs_normalises = scaler.fit_transform(indicateurs_eleves[['moyenne_annuelle', 'total_absences',
                                                            'total_retards', 'total_punitions',
                                                            'total_passages_infirmerie', 'sexe']])

    # Convertir en DataFrame pour ajouter la colonne 'cluster'
    indicateurs_normalises_df = pd.DataFrame(indicateurs_normalises, columns=['moyenne_annuelle', 'total_absences',
                                                                            'total_retards', 'total_punitions',
                                                                            'total_passages_infirmerie', 'sexe'])

    # Appliquer le clustering K-means avec la variable `sexe` incluse
    kmeans = KMeans(n_clusters=4, random_state=0)
    indicateurs_normalises_df['cluster'] = kmeans.fit_predict(indicateurs_normalises)

    # Ajouter les clusters au DataFrame d'origine
    indicateurs_eleves['cluster'] = indicateurs_normalises_df['cluster']

    # Calcul des pourcentages d'élèves dans chaque cluster
    profil_counts = indicateurs_eleves['cluster'].value_counts(normalize=True) * 100
    profil_counts.index = [f"Profil {i + 1}" for i in profil_counts.index]  # Renommer les index pour afficher "Profil 1", "Profil 2", etc.

    # Création du graphique en camembert avec Plotly
    fig_profil = px.pie(
        names=profil_counts.index,  # Utiliser les noms des profils pour chaque secteur
        values=profil_counts.values,  # Utiliser les pourcentages calculés pour chaque profil
        # title="Répartition des élèves par profil",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.G10
    )

    st.subheader('Profilage des élèves')
    st.write(
        "Ce graphique illustre la répartition des élèves en différents profils, déterminés par une analyse de clustering. Cette analyse a été réalisée en croisant la moyenne annuelle, les absences, les retards, les punitions, les passages à l'infirmerie, ainsi que le sexe des élèves. On peut identifier **cinq groupes** d’élèves présentant des caractéristiques communes.")

    col1, col2,col3, col4  =st.columns(4)

    with col1:
        with st.popover("Profil 1"):
            st.markdown("""
            Ce groupe se caractérise par une **moyenne annuelle de 12,01**, avec un nombre élevé d'**absences (31,95)** et de **retards (19,9)**. Les élèves de ce profil reçoivent également un nombre significatif de **punitions (10,06)** et ont un nombre modéré de passages à l'infirmerie (**4,24**). Ce profil inclut une majorité mixte avec une proportion de garçons de **53,76 %**. Ce profil pourrait correspondre à des élèves ayant des besoins en termes de suivi assiduité et comportement.
            """)
    with col2:
        with st.popover("Profil 2"):
            st.markdown("""
            Les élèves de ce profil présentent une **moyenne annuelle élevée de 15,14** et affichent des valeurs faibles en **absences (13,43)**, **retards (5,54)**, et **punitions (1,01)**. Les passages à l'infirmerie sont également bas (**2,92**). Ce groupe est exclusivement composé de filles, suggérant un profil d'élèves académiquement performantes et disciplinées.
            """)
    with col3:
        with st.popover("Profil 3"):
            st.markdown("""
            Les élèves de ce groupe affichent une **moyenne annuelle de 12,46**, avec un nombre modéré d'**absences (19,25)**, **retards (6,5)**, et **punitions (5,25)**. Cependant, le nombre de passages à l'infirmerie est exceptionnellement élevé (**108**). Ce groupe est composé de filles à **75 %**. Ce profil pourrait indiquer des élèves nécessitant une attention particulière en termes de santé.
            """)
    with col4:
        with st.popover("Profil 4"):
            st.markdown("""
            Ce profil présente une **moyenne annuelle de 14,27**, un faible nombre d'**absences (13,43)**, de **retards (4,5)**, et de **punitions (2,34)**. Les passages à l'infirmerie restent limités (**2,28**). Ce groupe est exclusivement composé de garçons, ce qui pourrait correspondre à un profil d'élèves académiquement performants et disciplinés.
            """)

    st.plotly_chart(fig_profil)

    ############ FACTEURS IMPACTANTS ############
    st.subheader('Analyse des facteurs impactants')

    st.write("L'analyse des relations entre différents indicateurs scolaires (moyenne annuelle, absences, retards, punitions, genre) permet d'identifier dans quelle mesure ces variables sont corrélées entre elles. Une corrélation indique qu’une variation dans un indicateur est associée à une variation dans un autre, ce qui peut révéler des tendances sous-jacentes dans les comportements ou les performances des élèves.")

    # Création des scatter plots pour chaque relation
    fig_corr = sp.make_subplots(rows=2, cols=3, subplot_titles=(
        "Moyenne vs Absences", "Moyenne vs Retards", "Moyenne vs Punitions",
        "Absences vs Retards", "Absences vs Punitions", "Retards vs Punitions"))

    # Moyenne vs Absences
    scatter1 = px.scatter(indicateurs_eleves, x='total_absences', y='moyenne_annuelle',
                        color_discrete_sequence=[px.colors.qualitative.G10[0]])
    for trace in scatter1.data:
        fig_corr.add_trace(trace, row=1, col=1)

    # Moyenne vs Retards
    scatter2 = px.scatter(indicateurs_eleves, x='total_retards', y='moyenne_annuelle',
                        color_discrete_sequence=[px.colors.qualitative.G10[1]])
    for trace in scatter2.data:
        fig_corr.add_trace(trace, row=1, col=2)

    # Moyenne vs Punitions
    scatter3 = px.scatter(indicateurs_eleves, x='total_punitions', y='moyenne_annuelle',
                        color_discrete_sequence=[px.colors.qualitative.G10[2]])
    for trace in scatter3.data:
        fig_corr.add_trace(trace, row=1, col=3)

    # Absences vs Retards
    scatter4 = px.scatter(indicateurs_eleves, x='total_absences', y='total_retards',
                        color_discrete_sequence=[px.colors.qualitative.G10[3]])
    for trace in scatter4.data:
        fig_corr.add_trace(trace, row=2, col=1)

    # Absences vs Punitions
    scatter5 = px.scatter(indicateurs_eleves, x='total_absences', y='total_punitions',
                        color_discrete_sequence=[px.colors.qualitative.G10[4]])
    for trace in scatter5.data:
        fig_corr.add_trace(trace, row=2, col=2)

    # Retards vs Punitions
    scatter6 = px.scatter(indicateurs_eleves, x='total_retards', y='total_punitions',
                        color_discrete_sequence=[px.colors.qualitative.G10[5]])
    for trace in scatter6.data:
        fig_corr.add_trace(trace, row=2, col=3)

    # Mettre à jour la disposition du graphique
    fig_corr.update_layout(height=800, width=1000)

    # Personnaliser l'apparence de la grille
    fig_corr.update_xaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.5)")
    fig_corr.update_yaxes(showgrid=True, gridwidth=1, gridcolor="rgba(200, 200, 200, 0.5)")

    # Affichage dans Streamlit
    st.plotly_chart(fig_corr)


    # Calcul des moyennes des indicateurs par sexe
    sexe_summary = indicateurs_eleves.groupby('sexe').mean().reset_index()
    sexe_summary['sexe'] = sexe_summary['sexe'].map({0: 'Filles', 1: 'Garçons'})  # Mapper les valeurs de sexe

    # List of indicators to plot
    indicators = ['moyenne_annuelle', 'total_absences', 'total_retards', 'total_punitions', 'total_passages_infirmerie']
    indicator_names = ['Moyenne annuelle', 'Absences', 'Retards', 'Punitions', 'Passages Infirmerie']

    # Transformation des données
    df_melted = sexe_summary.melt(id_vars='sexe', value_vars=indicators, var_name='Indicateur', value_name='Valeur')
    df_melted['Indicateur'] = df_melted['Indicateur'].replace(dict(zip(indicators, indicator_names)))

    # Création du graphique
    fig_sex_bar = px.bar(df_melted, x='sexe', y='Valeur', color='sexe',
                facet_col='Indicateur', facet_col_wrap=5, color_discrete_sequence=px.colors.qualitative.G10,
                labels={'sexe': 'Sexe', 'Valeur': 'Valeur moyenne', 'Indicateur': 'Indicateur'})

    # Mise à jour de la disposition
    fig_sex_bar.update_layout(
        showlegend=True,
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),  # Position de la légende
        height=500,  # Ajustement de la hauteur du graphique
        margin=dict(t=50, b=0, l=0, r=0)  # Marges pour un affichage optimal
    )
    fig_sex_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Ajustement des espacements entre les facettes
    fig_sex_bar.update_xaxes(matches=None, title=None,showticklabels=False)  # Enlève le titre "Sexe" redondant
    fig_sex_bar.update_yaxes(showgrid=True)

    # Affichage dans Streamlit


    # Calcul des moyennes des indicateurs par niveau
    niveau_summary = indicateurs_eleves.groupby('niveau').mean().reset_index()

    # Transformation des données
    df_melted_niveau = niveau_summary.melt(id_vars='niveau', value_vars=indicators, var_name='Indicateur', value_name='Valeur')
    df_melted_niveau['Indicateur'] = df_melted_niveau['Indicateur'].replace(dict(zip(indicators, indicator_names)))

    # Création du graphique
    fig_niveau_bar = px.bar(df_melted_niveau, x='niveau', y='Valeur', color='niveau',
                            facet_col='Indicateur', facet_col_wrap=5, color_discrete_sequence=[px.colors.qualitative.G10[2],px.colors.qualitative.G10[3]],
                            labels={'niveau': 'Niveau', 'Valeur': 'Valeur moyenne', 'Indicateur': 'Indicateur'})

    # Mise à jour de la disposition
    fig_niveau_bar.update_layout(
        showlegend=True,
        legend_title_text=None,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),  # Position de la légende
        height=500,  # Ajustement de la hauteur du graphique
        margin=dict(t=50, b=0, l=0, r=0)  # Marges pour un affichage optimal
    )
    fig_niveau_bar.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))

    # Ajustement des espacements entre les facettes
    fig_niveau_bar.update_xaxes(matches=None, title=None, showticklabels=False)  # Enlève le titre "Niveau" redondant
    fig_niveau_bar.update_yaxes(showgrid=True)


    st.plotly_chart(fig_sex_bar)
    # Affichage dans Streamlit
    st.plotly_chart(fig_niveau_bar)

    # Explication du graphique
    st.write("""

    1. **Moyenne vs Absences, Retards, Punitions** : Il y a  une tendance globale où un nombre élevé d’absences, de retards, ou de punitions est souvent associé à une baisse de la moyenne annuelle. Bien que ces relations ne soient pas strictement linéaires, on observe qu’un cumul de comportements problématiques semble affecter la moyenne.

    2. **Absences, Retards, et Punitions entre eux** : Il semble exister une corrélation modérée entre absences, retards et punitions, suggérant que les élèves en difficulté sur un de ces aspects sont souvent concernés par les autres.

    3. **Écarts entre filles et garçons** : Les garçons présentent un nombre moyen de punitions et de retards supérieur à celui des filles. À l'inverse, les filles affichent un nombre moyen d'absences plus élevé que les garçons. Les autres indicateurs, comme la moyenne annuelle et les passages à l'infirmerie, restent relativement proches entre les deux groupes. Ces écarts peuvent orienter vers des actions spécifiques selon les besoins observés pour chaque groupe.

    4. **Écarts entre collège et lycée** : Les élèves du lycée affichent un nombre moyen d'absences nettement plus élevé que ceux du lycée. En revanche, les indicateurs de retards, punitions, et passages à l'infirmerie sont plus faibles pour les élèves du lycée. Les moyennes annuelles restent relativement similaires entre les deux niveaux, indiquant des performances académiques comparables malgré ces différences comportementales.

    """)



with tab2:
    # 1. Moyenne par Élève et par Matière
    # Calcul des moyennes des notes par élève et par matière
    moyennes_matiere = df_notes_devoir.groupby(['eleve_id', 'matiere'])['note'].mean().reset_index()
    moyennes_matiere.rename(columns={'note': 'moyenne_matiere'}, inplace=True)

    # 2. Évolution des Résultats par Trimestre et par Genre
    # Calcul des moyennes trimestrielles par élève, par matière et par genre
    moyennes_trimestre = df_notes_devoir.groupby(['eleve_id', 'matiere', 'trimestre', 'sexe'])['note'].mean().reset_index()
    moyennes_trimestre.rename(columns={'note': 'moyenne_trimestrielle'}, inplace=True)

    # Visualisation de l'évolution des moyennes trimestrielles par matière et par genre
    fig_evolution = px.line(
        moyennes_trimestre, x='trimestre', y='moyenne_trimestrielle', color='sexe',
        facet_col='matiere', facet_col_wrap=3, markers=True,
        labels={'moyenne_trimestrielle': 'Moyenne Trimestrielle', 'trimestre': 'Trimestre'},
        title="Évolution des moyennes trimestrielles par matière et par genre"
    )
    fig_evolution.update_layout(height=600)
    fig_evolution.show()

    # 3. Répartition des Notes par Matière, Classe et Genre
    # Calcul des moyennes et écarts-types des notes par matière, classe, et sexe
    stats_matiere_classe_genre = df_notes_devoir.groupby(['matiere', 'classe', 'sexe']).agg(
        moyenne_note=('note', 'mean'),
        ecart_type_note=('note', 'std')
    ).reset_index()

    # Visualisation de la répartition des notes par matière, classe et genre
    fig_distribution = px.bar(
        stats_matiere_classe_genre, x='matiere', y='moyenne_note', color='sexe',
        facet_col='classe', error_y='ecart_type_note', barmode='group',
        labels={'moyenne_note': 'Moyenne des Notes', 'matiere': 'Matière'},
        title="Répartition des moyennes par matière, classe et genre"
    )
    fig_distribution.update_layout(height=700)
    fig_distribution.show()

    # 4. Clustering par Matières
    # Préparation des données pour le clustering : Moyenne par matière pour chaque élève
    moyennes_par_matiere = df_notes_devoir.pivot_table(index='eleve_id', columns='matiere', values='note', aggfunc='mean').fillna(0)

    # Appliquer le clustering K-means
    kmeans = KMeans(n_clusters=4, random_state=0)
    moyennes_par_matiere['cluster'] = kmeans.fit_predict(moyennes_par_matiere)

    # Visualisation des clusters par matière (en utilisant les deux premières matières principales pour les axes)
    matiere_x = moyennes_par_matiere.columns[0]  # Première matière
    matiere_y = moyennes_par_matiere.columns[1]  # Deuxième matière

    fig_clusters = px.scatter(
        moyennes_par_matiere,
        x=matiere_x,
        y=matiere_y,
        color='cluster',
        title="Clusters d'élèves par matières",
        labels={matiere_x: f"Moyenne en {matiere_x}", matiere_y: f"Moyenne en {matiere_y}", 'cluster': 'Cluster'},
        color_continuous_scale=px.colors.qualitative.G10
    )
    fig_clusters.update_layout(height=600)
    fig_clusters.show()
