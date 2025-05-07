import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

def prepare_features_target(data, features, target):
    X = data[features].select_dtypes(include=[np.number])
    y = data[target]
    return X, y

def train_and_evaluate_model(X_train, y_train, X_val, y_val):
    # Initialisation et entraînement du modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble de validation
    y_pred = model.predict(X_val)
    
    # Calcul des métriques d'évaluation
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"MAE: {mae}\nMSE: {mse}\nRMSE: {rmse}")
    
    return model

def test_model(model, X_test, y_test):
    # Prédictions sur l'ensemble de test
    y_test_pred = model.predict(X_test)
    
    # Calcul des métriques
    mae_test = mean_absolute_error(y_test, y_test_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = np.sqrt(mse_test)
    
    print(f"\nPerformance sur le test:\nMAE: {mae_test}\nMSE: {mse_test}\nRMSE: {rmse_test}")

if __name__ == "__main__":
    data_dir = "data"
    models_dir = "models"
    
    # Créer le répertoire models s'il n'existe pas
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Charger les données prétraitées
    train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
    val_data = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
    test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
    
    # Définition des variables
    features = [
        "minimum_temperature_at_2_metres", "maximum_temperature_at_2_metres",
        "2_metre_relative_humidity", "total_precipitation", "10m_wind_speed",
        "surface_net_solar_radiation", "surface_net_thermal_radiation",
        "surface_solar_radiation_downwards", "surface_latent_heat_flux"
    ]
    
    # Liste des cibles à prédire
    targets = [
        "2_metre_temperature",
        "2_metre_dewpoint_temperature",
        "total_precipitation",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
    ]
    
    # Entraîner un modèle pour chaque cible
    for target in targets:
        print(f"\n{'='*50}")
        print(f"Entraînement du modèle pour la cible: {target}")
        print(f"{'='*50}")
        
        # Vérifier que la cible existe dans les données
        if target not in train_data.columns:
            print(f"ERREUR: La cible '{target}' n'existe pas dans les données d'entraînement.")
            continue
            
        # Préparer les données
        X_train, y_train = prepare_features_target(train_data, features, target)
        X_val, y_val = prepare_features_target(val_data, features, target)
        X_test, y_test = prepare_features_target(test_data, features, target)
        
        # Entraîner et évaluer le modèle
        model = train_and_evaluate_model(X_train, y_train, X_val, y_val)
        
        # Tester le modèle
        test_model(model, X_test, y_test)
        
        # Sauvegarder le modèle
        model_filename = f"meteo_model_{target.replace(' ', '_')}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé dans {model_path}")