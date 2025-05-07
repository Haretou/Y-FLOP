# train_model_simple.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os
import pickle

def prepare_features_target(data, features, target):
    X = data[features].select_dtypes(include=[np.number])
    X = X.fillna(X.mean())
    y = data[target]
    if y.isnull().any():
        y = y.fillna(y.mean())
    return X, y

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    # Seulement LinearRegression pour tester
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de validation
    y_pred = model.predict(X_val)
    
    # Calcul des métriques d'évaluation
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}")
    
    return model

if __name__ == "__main__":
    data_dir = "data"
    models_dir = "models"
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Charger les données prétraitées
    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    val_data = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
    
    # Définition des features de base (sans les nouvelles caractéristiques)
    features = [
        "minimum_temperature_at_2_metres", "maximum_temperature_at_2_metres",
        "2_metre_relative_humidity", "total_precipitation", "10m_wind_speed",
        "surface_net_solar_radiation", "surface_net_thermal_radiation", 
        "surface_solar_radiation_downwards", "surface_latent_heat_flux"
    ]
    
    # Vérifier les caractéristiques présentes
    available_features = []
    for feature in features:
        if feature in train_data.columns:
            available_features.append(feature)
    
    print(f"Features disponibles: {available_features}")
    
    # Cible à prédire
    target = "2_metre_temperature"
    
    # Préparer les données
    X_train, y_train = prepare_features_target(train_data, available_features, target)
    X_val, y_val = prepare_features_target(val_data, available_features, target)
    
    # Entraîner et évaluer le modèle
    model = train_and_evaluate_model(X_train, y_train, X_val, y_val)
    
    # Sauvegarder le modèle dans un format simple
    model_filename = os.path.join(models_dir, "meteo_model.pkl")
    joblib.dump(model, model_filename)
    print(f"Modèle sauvegardé dans {model_filename}")