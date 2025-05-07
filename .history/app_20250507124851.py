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
        'lat': lat,
        'lon': lon
    }
    
    return jsonify(weather_data)

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer les données du formulaire
    form_data = {}
    for feature in features:
        try:
            form_data[feature] = float(request.form.get(feature, 0))
        except (ValueError, TypeError):
            # En cas d'erreur, utiliser une valeur par défaut
            form_data[feature] = 0
    
    # Créer un DataFrame avec les données
    input_data = pd.DataFrame([form_data])
    
    # Standardiser les données si un scaler est disponible
    if 'scaler' in globals() and scaler is not None:
        input_data_scaled = pd.DataFrame(
            scaler.transform(input_data),
            columns=input_data.columns
        )
    else:
        input_data_scaled = input_data
    
    # Faire la prédiction avec le modèle principal
    prediction = model.predict(input_data_scaled)[0]
    
    # Appliquer le correcteur d'erreur si disponible
    if 'error_corrector' in globals() and error_corrector is not None:
        error_prediction = error_corrector.predict(input_data_scaled)[0]
        prediction += error_prediction
    
    # Utiliser l'ensemble si disponible
    if 'ensemble_models' in globals() and ensemble_models is not None:
        ensemble_predictions = [model.predict(input_data_scaled)[0] for model in ensemble_models]
        ensemble_prediction = np.mean(ensemble_predictions)
        # Pondérer entre la prédiction principale et l'ensemble
        prediction = 0.7 * prediction + 0.3 * ensemble_prediction
    
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
        
        # Utiliser les valeurs min et max observées comme limites pour les prévisions
        min_temp = commune_data['2_metre_temperature'].min()
        max_temp = commune_data['2_metre_temperature'].max()
        
        # Extraire la latitude et longitude pour ajuster les prévisions
        lat, lon = None, None
        if 'position' in commune_data.columns:
            position = commune_data['position'].iloc[0]
            position_match = re.search(r'([\d\.-]+),\s*([\d\.-]+)', str(position))
            if position_match:
                lat, lon = float(position_match.group(1)), float(position_match.group(2))
    else:
        base_temp = 10  # Valeur par défaut
        min_temp, max_temp = 5, 25  # Limites par défaut
        lat, lon = None, None
    
    # Simuler des prévisions plus réalistes pour les prochains jours
    days = 5
    forecasts = []
    
    # Paramètres pour des fluctuations réalistes
    day_variation = 3.0  # Variation jour-jour
    max_jump = 1.5  # Saut maximum entre deux jours
    smoothing_factor = 0.7  # Pour lisser les transitions
    
    # Température du jour précédent (commence avec la base)
    prev_temp = base_temp
    
    for i in range(days):
        date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
        
        # Simuler une tendance saisonnière (plus chaud en été, plus froid en hiver)
        month = datetime.now().month
        seasonal_factor = np.sin(2 * np.pi * (month - 1) / 12)  # -1 à +1
        
        # Utiliser le modèle pour prédire si possible
        if model_components is not None and commune_name and lat is not None and lon is not None:
            try:
                # Créer des données d'entrée appropriées pour le modèle
                predict_data = {
                    "minimum_temperature_at_2_metres": max(min_temp, base_temp - day_variation + seasonal_factor),
                    "maximum_temperature_at_2_metres": min(max_temp, base_temp + day_variation + seasonal_factor),
                    "2_metre_relative_humidity": 60 + np.random.normal(0, 10),
                    "total_precipitation": max(0, np.random.exponential(2) if np.random.random() < 0.3 else 0),
                    "10m_wind_speed": 5 + np.random.normal(0, 2)
                }
                
                # Ajouter les autres features requises avec des valeurs par défaut
                for feature in features:
                    if feature not in predict_data:
                        predict_data[feature] = 0
                
                # Standardiser et prédire
                input_df = pd.DataFrame([predict_data])
                input_scaled = pd.DataFrame(
                    scaler.transform(input_df),
                    columns=input_df.columns
                )
                
                # Faire la prédiction
                temp_prediction = model.predict(input_scaled)[0]
                
                # Appliquer la correction si disponible
                if error_corrector is not None:
                    temp_prediction += error_corrector.predict(input_scaled)[0]
                
                # Lisser avec la température précédente pour éviter les sauts irréalistes
                temp = smoothing_factor * prev_temp + (1 - smoothing_factor) * temp_prediction
                
                # Limiter les variations extrêmes
                temp = min(max(temp, prev_temp - max_jump), prev_temp + max_jump)
                
            except Exception as e:
                print(f"Erreur lors de la prédiction pour la prévision: {e}")
                # Fallback à la méthode de simulation
                temp_variation = np.random.normal(0, day_variation / 2)
                temp = prev_temp + temp_variation
                # Limiter les variations extrêmes
                temp = min(max(temp, prev_temp - max_jump), prev_temp + max_jump)
        else:
            # Méthode de simulation simple
            temp_variation = np.random.normal(0, day_variation / 2)
            temp = prev_temp + temp_variation
            # Limiter les variations extrêmes
            temp = min(max(temp, prev_temp - max_jump), prev_temp + max_jump)
        
        # Mettre à jour la température précédente
        prev_temp = temp
        
        # Calculer l'humidité en tenant compte de la température
        # L'humidité tend à être plus basse quand il fait plus chaud
        humidity_base = 70 - (temp - base_temp) * 2
        humidity = max(30, min(95, humidity_base + np.random.normal(0, 7)))
        
        # Précipitations liées à l'humidité (plus probable quand humidité élevée)
        precip_prob = (humidity - 50) / 100 if humidity > 50 else 0
        precipitation = np.random.exponential(3) if np.random.random() < precip_prob else 0
        precipitation = max(0, min(40, precipitation))  # Limiter les valeurs extrêmes
        
        forecasts.append({
            'date': date,
            'temperature': round(temp, 1),
            'humidity': round(humidity, 1),
            'precipitation': round(precipitation, 1)
        })
    
    return jsonify(forecasts)

@app.route('/model_info')
def model_info():
    """Fournit des informations sur le modèle actuel"""
    if model_components is not None:
        info = {
            'target': model_components.get('target', 'inconnu'),
            'features_count': len(model_components.get('features', [])),
            'features': model_components.get('features', []),
            'model_type': type(model_components.get('main_model', None)).__name__,
            'error_correction': error_corrector is not None,
            'ensemble_model': ensemble_models is not None,
            'metrics': model_components.get('metrics', {})
        }
    else:
        info = {
            'model_type': type(model).__name__,
            'features_count': len(features),
            'features': features
        }
    
    return jsonify(info)

if __name__ == '__main__':
    app.run(debug=True)