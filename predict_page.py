import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Charger le modèle pré-entraîné
model = joblib.load("model.pkl")
loaded_scaler = joblib.load('scaler.pkl')
# To load the encoder from the file
le_for_adres =joblib.load('LabelEncoder_adress.pkl')
le_for_city =joblib.load('LabelEncoder_city.pkl')

# Créer une interface utilisateur Streamlit
st.title("Prédiction de la valeur foncière d'un bien immobilier")
st.write("Bienvenue dans notre outil de prédiction de la valeur foncière.")

# Formulaire pour saisir les caractéristiques du bien immobilier
st.header("Caractéristiques du bien immobilier necessaires pour la prédiction")
adresse_nom_voie = st.text_input("Adresse Nom Voie",value='RUE DE LA CHARPINE')
adresse_numero = st.number_input("Numéro de l'adresse", value=843)
nom_commune = st.text_input("Nom de la ville",value='Simandre-sur-Suran')
code_postal = st.number_input("Code postal", value=1000)
type_local = st.selectbox("Type de local", ['Maison', 'Appartement'])
surface_reelle_bati = st.number_input("Surface", value=0)
nombre_pieces_principales = st.number_input("Nombre de pièces principales", value=0, step=1)



st.header("Information optionnelle ")
constructionYear = st.number_input("Année de construction", value=2017)

# Bouton pour effectuer la prédiction
if st.button("Prédire la valeur foncière"):
    
    df = pd.DataFrame({
        'adresse_nom_voie': [adresse_nom_voie],
        'adresse_numero': [adresse_numero],
        'nom_commune': [nom_commune],
        'code_postal': [code_postal],
        'surface_reelle_bati': [surface_reelle_bati],
        'nombre_pieces_principales': [nombre_pieces_principales],
        'constructionYear': [constructionYear],
        'type_local': [type_local],
 
    })
    df['type_local'] = df['type_local'].replace({"Appartement": 1, "Maison": 2})
    df['adresse_nom_voie']=le_for_adres.fit_transform( df['adresse_nom_voie'])
    df['nom_commune']=le_for_city.fit_transform( df['nom_commune'])
    

    # Reset the indices of both DataFrames
    

    
    
    X_test = df.values
    X_test = loaded_scaler.transform(X_test)
    prediction = model.predict(X_test, num_iteration=model.best_iteration)
    #prediction = np.expm1( model.predict(X_test, num_iteration=model.best_iteration)) 
    
    st.subheader("Résultat de la prédiction")
    st.write(f"La valeur foncière prédite est : { int(prediction[0])} euros")

# Astuce : Vous pouvez personnaliser davantage votre interface utilisateur Streamlit en ajoutant des graphiques, des informations supplémentaires, etc.
