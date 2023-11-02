import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle pré-entraîné
model = joblib.load("model.pkl")
loaded_scaler = joblib.load('scaler.pkl')
# To load the encoder from the file
loaded_encoder = joblib.load("OneHotEncoder.pkl")
le_for_adres =joblib.load('LabelEncoder_adress.pkl')
le_for_nature =joblib.load('LabelEncoder_nature.pkl')

# Créer une interface utilisateur Streamlit
st.title("Prédiction de la valeur foncière d'un bien immobilier")
st.write("Bienvenue dans notre outil de prédiction de la valeur foncière.")

# Formulaire pour saisir les caractéristiques du bien immobilier
st.header("Caractéristiques du bien immobilier")
adresse_nom_voie = st.text_input("Adresse Nom Voie",value='RUE DE LA CHARPINE')
code_departement = st.number_input("Code du département", value=1)
code_commune = st.number_input("Code de la commune", value=1430)
numero_disposition = st.number_input("Numéro de disposition", value=1)
adresse_numero = st.number_input("Numéro de l'adresse", value=843)
code_postal = st.number_input("Code postal", value=1000)
nombre_lots = st.number_input("Nombre de lots", value=0, step=1)
code_type_local = st.selectbox("Code du type de local", [1, 2, 3, 4])
surface_reelle_bati = st.number_input("Surface réelle du bâti", value=73)
nombre_pieces_principales = st.number_input("Nombre de pièces principales", value=0, step=1)
surface_terrain = st.number_input("Surface du terrain", value=391.0)
longitude = st.number_input("Longitude", value=5.204386)
latitude = st.number_input("Latitude", value=6.193683)

# Formulaire pour saisir les informations sur la mutation
st.header("Informations sur la mutation")
date_mutation = st.date_input("Date de la mutation", pd.to_datetime("2017-07-04"))
nature_mutation = st.selectbox("Nature de la mutation", ['Vente', 'Vente terrain à bâtir', 'Echange', "Vente en l'état futur d'achèvement", 'Adjudication', 'Expropriation'])
type_local = st.selectbox("Type de local", ['Maison', 'Appartement', 'Local industriel. commercial ou assimilé'])
nature_culture = st.selectbox("Nature de la culture", ['sols', 'terrains à bâtir', "terrains d'agrément", 'terres', 'prés', 'taillis sous futaie', 'jardins', 'vignes', 'carrières', 'landes', 'taillis simples', 'peupleraies', 'eaux', 'vergers', 'futaies résineuses', 'futaies mixtes', 'pâtures', 'futaies feuillues', 'chemin de fer', 'bois', 'prés plantes', 'pacages', 'oseraies', 'terres plantées', 'landes boisées', 'herbages', "prés d'embouche"])

# Bouton pour effectuer la prédiction
if st.button("Prédire la valeur foncière"):
    
    df = pd.DataFrame({
        'adresse_nom_voie': [adresse_nom_voie],
        'code_departement': [code_departement],
        'code_commune': [code_commune],
        'numero_disposition': [numero_disposition],
        'adresse_numero': [adresse_numero],
        'code_postal': [code_postal],
        'nombre_lots': [nombre_lots],
        'code_type_local': [code_type_local],
        'surface_reelle_bati': [surface_reelle_bati],
        'nombre_pieces_principales': [nombre_pieces_principales],
        'surface_terrain': [surface_terrain],
        'longitude': [longitude],
        'latitude': [latitude],
        'date_mutation': [date_mutation],
        'nature_mutation': [nature_mutation],
        'type_local': [type_local],
        'nature_culture': [nature_culture]
    })
    print(df)
    print('-------------------------------------------')
    adres = le_for_adres.fit_transform( df['adresse_nom_voie'])
    nature = le_for_nature.fit_transform( df['nature_culture'])

    
    adres = pd.DataFrame(adres, columns=['adresse_nom_voie'])
    nature = pd.DataFrame(nature, columns=['nature_culture'])
    multiple_enc=loaded_encoder.transform(df[['nature_mutation', 'type_local']])
    multiple_enc=multiple_enc.toarray()
    # Use oenc.get_feature_names() to get the column names of the one-hot encoded features
    multiple_enc = pd.DataFrame(multiple_enc, columns=['x0_Echange', 'x0_Expropriation', 'x0_Vente',
       "x0_Vente en l'état futur d'achèvement",
       'x0_Vente terrain à bâtir',
       'x1_Local industriel. commercial ou assimilé', 'x1_Maison'])

    # Reset the indices of both DataFrames
    df = df.reset_index(drop=True)
    multiple_enc = multiple_enc.reset_index(drop=True)

    numerical_cols = [f for f in df.columns if df.dtypes[f] != 'object']
    # Concatenate the DataFrames along axis 1
    df_wo_outliers_E = pd.concat([df[numerical_cols], multiple_enc,adres,nature], axis=1,ignore_index=True)
    df_wo_outliers_E.columns = numerical_cols + list(multiple_enc.columns)+['adresse_nom_voie','nature_culture']
    df['date_mutation'] = pd.to_datetime(df['date_mutation'])
    df_wo_outliers_E['month'] = df['date_mutation'].dt.month
    df_wo_outliers_E['day'] = df['date_mutation'].dt.day
    df_wo_outliers_E['year'] = df['date_mutation'].dt.year
    X_test = df_wo_outliers_E.values
    X_test = loaded_scaler.transform(X_test)
    prediction = model.predict(X_test, num_iteration=model.best_iteration)
    #prediction = np.expm1( model.predict(X_test, num_iteration=model.best_iteration)) 
    
    st.subheader("Résultat de la prédiction")
    st.write(f"La valeur foncière prédite est : { int(prediction[0])} euros")

# Astuce : Vous pouvez personnaliser davantage votre interface utilisateur Streamlit en ajoutant des graphiques, des informations supplémentaires, etc.
