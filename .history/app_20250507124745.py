# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import re
import pickle
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Charger le modèle avec tous ses composants
def load_model_components(target="2_metre_temperature"):
    model_path = os.path.join('models', f'meteo_model_{target}.pkl')
    try:
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
        return model_components
    except (FileNotFoundError, pickle.UnpicklingError) as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None

# Charger le modèle principal
model_components = load_model_components()

# Si le modèle n'est pas disponible, utiliser un modèle de secours
if model_components is None:
    # Chercher un modèle plus ancien au format .pkl (pour la compatibilité)
    backup_model_path = os.path.join('models', 'meteo_model.pkl')
    if os.path.exists(backup_model_path):
        model = joblib.load(backup_model_path)
        features = [
            "minimum_temperature_at_2_metres", "maximum_temperature_at_2_metres",
            "2_metre_relative_humidity", "total_precipitation", "10m_wind_speed",
            "surface_net_solar_radiation", "surface_net_thermal_radiation",
            "surface_solar_radiation_downwards", "surface_latent_heat_flux"
        ]
    else:
        raise RuntimeError("Aucun modèle disponible! Exécutez d'abord preprocessing.py et train_model.py.")
else:
    # Extraire les composants du modèle
    features = model_components['features']
    model = model_components['main_model']
    scaler = model_components['scaler']
    error_corrector = model_components.get('error_corrector')
    ensemble_models = model_components.get('ensemble_models')

# Charger le fichier CSV des données météo
meteo_path = os.path.join('data', 'val_data.csv')
meteo_df = pd.read_csv(meteo_path, delimiter=',')

# Extraire la liste unique des communes
communes_df = meteo_df[['commune', 'code_commune']].drop_duplicates().sort_values('commune')

@app.route('/')
def index():
    # Envoyer la liste des communes au template
    communes = communes_df['commune'].tolist()
    return render_template('index.html', cities=communes)

@app.route('/search_city', methods=['GET'])
def search_city():
    # Récupérer le terme de recherche
    query = request.args.get('q', '').lower()
    
    # Filtrer les communes qui correspondent à la recherche
    matching_communes = communes_df[communes_df['commune'].str.lower().str.contains(query)].to_dict('records')
    
    return jsonify(matching_communes)

@app.route('/city_weather/<commune_name>')
def city_weather(commune_name):
    # Trouver les données météo récentes pour cette commune
    commune_data = meteo_df[meteo_df['commune'] == commune_name]
    
    if commune_data.empty:
        return jsonify({'error': 'Commune non trouvée'}), 404
    
    # Trier par timestamp pour obtenir les données les plus récentes
    commune_data = commune_data.sort_values('forecast_timestamp', ascending=False).iloc[0]
    
    # Extraire les coordonnées de la position (format "lat, lon")
    lat, lon = None, None
    if 'position' in commune_data:
        position_match = re.search(r'([\d\.-]+),\s*([\d\.-]+)', commune_data['position'])
        if position_match:
            lat, lon = float(position_match.group(1)), float(position_match.group(2))
    
    # Préparer les données météo
    weather_data = {
        'commune': commune_name,
        'code_commune': commune_data['code_commune'],
        'temperature': round(commune_data['2_metre_temperature'], 1),
        'humidity': round(commune_data['2_metre_relative_humidity'], 1),
        'precipitation': round(commune_data['total_precipitation'], 1),
        'wind_speed': round(commune_data['10m_wind_speed'], 1),
        'timestamp': commune_data['forecast_timestamp'],
        'lat