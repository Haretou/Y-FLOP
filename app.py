# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime, timedelta

app = Flask(__name__)

# Charger le modèle
model_path = os.path.join('models', 'meteo_model.pkl')
model = joblib.load(model_path)

# Liste des fonctionnalités nécessaires pour la prédiction
features = [
    "minimum_temperature_at_2_metres", "maximum_temperature_at_2_metres",
    "2_metre_relative_humidity", "total_precipitation", "10m_wind_speed",
    "surface_net_solar_radiation", "surface_net_thermal_radiation",
    "surface_solar_radiation_downwards", "surface_latent_heat_flux"
]

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
        'lat': lat,
        'lon': lon
    }
    
    return jsonify(weather_data)

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    data = {}
    for feature in features:
        data[feature] = float(request.form.get(feature, 0))
    
    # Créer un DataFrame avec les données
    input_data = pd.DataFrame([data])
    
    # Faire la prédiction
    prediction = model.predict(input_data)[0]
    
    return jsonify({
        'temperature': round(prediction, 2),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/forecast/<commune_name>')
def forecast(commune_name=None):
    # Si une commune est spécifiée, filtrer les données pour cette commune
    if commune_name:
        commune_data = meteo_df[meteo_df['commune'] == commune_name]
        if commune_data.empty:
            return jsonify({'error': 'Commune non trouvée'}), 404
            
        # Trouver la température moyenne pour cette commune comme base
        base_temp = commune_data['2_metre_temperature'].mean()
    else:
        base_temp = 10  # Valeur par défaut
    
    # Simuler des prévisions pour les prochains jours
    # Dans un cas réel, vous utiliseriez les données disponibles dans votre CSV
    days = 5
    forecasts = []
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        temp = base_temp + np.random.normal(0, 2)  # Température avec variation aléatoire
        forecasts.append({
            'date': date,
            'temperature': round(temp, 1),
            'humidity': round(50 + np.random.normal(0, 10), 1),
            'precipitation': max(0, round(np.random.normal(0, 5), 1))
        })
    
    return jsonify(forecasts)

if __name__ == '__main__':
    app.run(debug=True)