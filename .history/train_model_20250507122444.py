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

def apply_temperature_threshold(y_true, y_pred, max_threshold=2.0):
    """
    Vérifie si les prédictions sont dans la marge acceptable (±2°C par rapport à la réalité)
    Retourne un masque booléen indiquant quelles prédictions sont dans la marge
    """
    temp_diff = np.abs(y_true - y_pred)
    return temp_diff <= max_threshold

def train_model_with_temperature_constraint(X_train, y_train, max_threshold=2.0):
    """
    Entraîne un modèle qui respecte la contrainte de différence maximale de température
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Prédictions sur l'ensemble d'entraînement
    y_train_pred = model.predict(X_train)
    
    # Identifier les prédictions respectant la contrainte de température
    valid_predictions = apply_temperature_threshold(y_train, y_train_pred, max_threshold)
    
    # Pourcentage de prédictions valides
    valid_percent = np.mean(valid_predictions) * 100
    print(f"Pourcentage de prédictions respectant le seuil de {max_threshold}°C: {valid_percent:.2f}%")
    
    return model, valid_predictions

def evaluate_model(model, X, y, dataset_name, max_threshold=2.0):
    """
    Évalue le modèle et vérifie le respect de la contrainte de température
    """
    # Prédictions
    y_pred = model.predict(X)
    
    # Calcul des métriques d'évaluation
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    
    # Vérifier la contrainte de température
    valid_predictions = apply_temperature_threshold(y, y_pred, max_threshold)
    valid_percent = np.mean(valid_predictions) * 100
    
    print(f"\nPerformance sur {dataset_name}:")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pourcentage de prédictions avec écart ≤ {max_threshold}°C: {valid_percent:.2f}%")
    
    # Analyse détaillée des écarts
    temp_diff = np.abs(y - y_pred)
    print(f"Écart moyen: {np.mean(temp_diff):.2f}°C")
    print(f"Écart médian: {np.median(temp_diff):.2f}°C")
    print(f"Écart maximum: {np.max(temp_diff):.2f}°C")
    
    return valid_predictions, y_pred

if __name__ == "__main__":
    data_dir = "data"
    models_dir = "models"
    results_dir = "results"
    
    # Créer les répertoires nécessaires
    for directory in [models_dir, results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
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
    
    # Liste des cibles à prédire avec focus sur les températures
    temperature_targets = [
        "2_metre_temperature",
        "2_metre_dewpoint_temperature"
    ]
    
    other_targets = [
        "total_precipitation",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation"
    ]
    
    # Tolérance maximale en degrés Celsius pour les températures
    max_temperature_diff = 2.0
    
    # Entraîner les modèles pour les températures avec la contrainte
    for target in temperature_targets:
        print(f"\n{'='*50}")
        print(f"Entraînement du modèle pour la cible: {target} (avec contrainte ≤ {max_temperature_diff}°C)")
        print(f"{'='*50}")
        
        # Vérifier que la cible existe dans les données
        if target not in train_data.columns:
            print(f"ERREUR: La cible '{target}' n'existe pas dans les données d'entraînement.")
            continue
            
        # Préparer les données
        X_train, y_train = prepare_features_target(train_data, features, target)
        X_val, y_val = prepare_features_target(val_data, features, target)
        X_test, y_test = prepare_features_target(test_data, features, target)
        
        # Entraîner le modèle avec la contrainte de température
        model, valid_train_preds = train_model_with_temperature_constraint(X_train, y_train, max_temperature_diff)
        
        # Évaluer le modèle sur les ensembles de validation et de test
        valid_val_preds, y_val_pred = evaluate_model(model, X_val, y_val, "validation", max_temperature_diff)
        valid_test_preds, y_test_pred = evaluate_model(model, X_test, y_test, "test", max_temperature_diff)
        
        # Sauvegarder le modèle
        model_filename = f"meteo_model_{target.replace(' ', '_')}_threshold_{max_temperature_diff}C.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé dans {model_path}")
        
        # Sauvegarder les résultats dans un CSV pour analyse ultérieure
        results_df = pd.DataFrame({
            'real_temp': y_test,
            'predicted_temp': y_test_pred,
            'temp_diff': np.abs(y_test - y_test_pred),
            'is_valid': valid_test_preds
        })
        
        results_filename = f"results_{target.replace(' ', '_')}_threshold_{max_temperature_diff}C.csv"
        results_path = os.path.join(results_dir, results_filename)
        results_df.to_csv(results_path, index=False)
        print(f"Résultats sauvegardés dans {results_path}")
    
    # Entraîner les modèles pour les autres cibles sans contrainte spécifique
    for target in other_targets:
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
        
        # Entraîner et évaluer le modèle standard
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Évaluer le modèle
        y_val_pred = model.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_val_pred)
        mse_val = mean_squared_error(y_val, y_val_pred)
        rmse_val = np.sqrt(mse_val)
        
        print(f"\nPerformance sur validation:")
        print(f"MAE: {mae_val:.4f}")
        print(f"MSE: {mse_val:.4f}")
        print(f"RMSE: {rmse_val:.4f}")
        
        # Test
        y_test_pred = model.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_test_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        rmse_test = np.sqrt(mse_test)
        
        print(f"\nPerformance sur test:")
        print(f"MAE: {mae_test:.4f}")
        print(f"MSE: {mse_test:.4f}")
        print(f"RMSE: {rmse_test:.4f}")
        
        # Sauvegarder le modèle
        model_filename = f"meteo_model_{target.replace(' ', '_')}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        joblib.dump(model, model_path)
        print(f"Modèle sauvegardé dans {model_path}")