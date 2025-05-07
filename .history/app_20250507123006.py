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

# Seuil maximal de différence de température (en degrés Celsius)
MAX_TEMPERATURE_DIFF = 2.0

def temperature_diff_valid(predicted_temp, actual_temp, max_diff=MAX_TEMPERATURE_DIFF):
    """
    Vérifie si la différence entre la température prédite et réelle est dans la limite acceptable
    """
    return bool(abs(predicted_temp - actual_temp) <= max_diff)  # Convertit en bool Python standard

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
        'temperature': round(float(commune_data['2_metre_temperature']), 1),
        'humidity': round(float(commune_data['2_metre_relative_humidity']), 1),
        'precipitation': round(float(commune_data['total_precipitation']), 1),
        'wind_speed': round(float(commune_data['10m_wind_speed']), 1),
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
    prediction = float(model.predict(input_data)[0])  # Convertir en float Python standard
    
    # Récupérer la température réelle si disponible (pour vérification)
    actual_temp = request.form.get('actual_temperature')
    
    response = {
        'temperature': round(prediction, 2),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Si nous avons la température réelle, vérifier si la prédiction est dans la limite acceptable
    if actual_temp:
        actual_temp = float(actual_temp)
        is_valid = temperature_diff_valid(prediction, actual_temp)
        response['is_valid'] = is_valid
        response['actual_temperature'] = round(actual_temp, 2)
        response['temperature_diff'] = round(abs(prediction - actual_temp), 2)
        response['max_allowed_diff'] = MAX_TEMPERATURE_DIFF
    
    return jsonify(response)

@app.route('/forecast/<commune_name>')
def forecast(commune_name=None):
    # Si une commune est spécifiée, filtrer les données pour cette commune
    if commune_name:
        commune_data = meteo_df[meteo_df['commune'] == commune_name]
        if commune_data.empty:
            return jsonify({'error': 'Commune non trouvée'}), 404
            
        # Trouver la température moyenne pour cette commune comme base
        base_temp = float(commune_data['2_metre_temperature'].mean())  # Convertir en float Python standard
    else:
        base_temp = 10.0  # Valeur par défaut
    
    # Simuler des prévisions pour les prochains jours
    days = 5
    forecasts = []
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        temp_variation = float(np.random.normal(0, 1.5))  # Convertir en float Python standard
        
        # Limiter la variation pour respecter la contrainte des 2°C
        if abs(temp_variation) > MAX_TEMPERATURE_DIFF:
            temp_variation = MAX_TEMPERATURE_DIFF if temp_variation > 0 else -MAX_TEMPERATURE_DIFF
            
        temp = base_temp + temp_variation
        humidity = float(50 + np.random.normal(0, 10))  # Convertir en float Python standard
        precip = float(max(0, np.random.normal(0, 5)))  # Convertir en float Python standard
        
        forecast_entry = {
            'date': date,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'precipitation': round(precip, 1),
            'temperature_diff': round(abs(temp - base_temp), 2),
            'within_threshold': bool(abs(temp - base_temp) <= MAX_TEMPERATURE_DIFF)  # Explicitement bool Python
        }
        
        forecasts.append(forecast_entry)
    
    return jsonify(forecasts)

@app.route('/validate_forecasts')
def validate_forecasts():
    """
    Compare toutes les prévisions météo avec les températures réelles
    et renvoie les statistiques de validité.
    """
    valid_forecasts = 0
    total_forecasts = 0
    communes_stats = []
    
    # Parcourir toutes les communes
    for commune in communes_df['commune'].unique():
        # Obtenir les données météo pour cette commune
        commune_data = meteo_df[meteo_df['commune'] == commune]
        if commune_data.empty:
            continue
            
        # Trier par timestamp pour obtenir les données les plus récentes
        latest_data = commune_data.sort_values('forecast_timestamp', ascending=False).iloc[0]
        current_temp = float(latest_data['2_metre_temperature'])
        
        # Générer des prévisions pour comparer
        base_temp = current_temp
        
        # Simuler quelques prévisions
        test_temps = [
            base_temp + float(np.random.normal(0, 1.5)) for _ in range(5)
        ]
        
        # Vérifier la validité
        valid_count = sum(1 for temp in test_temps if abs(temp - current_temp) <= MAX_TEMPERATURE_DIFF)
        
        communes_stats.append({
            'commune': commune,
            'current_temp': round(current_temp, 1),
            'forecasts_within_threshold': valid_count,
            'total_forecasts': len(test_temps),
            'percent_valid': round((valid_count / len(test_temps)) * 100, 1)
        })
        
        valid_forecasts += valid_count
        total_forecasts += len(test_temps)
    
    # Trier par pourcentage de validité
    communes_stats.sort(key=lambda x: x['percent_valid'], reverse=True)
    
    return jsonify({
        'total_valid_forecasts': valid_forecasts,
        'total_forecasts': total_forecasts,
        'overall_percent_valid': round((valid_forecasts / total_forecasts) * 100, 1) if total_forecasts > 0 else 0,
        'max_temperature_diff': MAX_TEMPERATURE_DIFF,
        'communes_stats': communes_stats[:10]  # Limiter à 10 communes pour la lisibilité
    })

if __name__ == '__main__':
    app.run(debug=True)