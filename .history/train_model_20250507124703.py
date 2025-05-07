# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time
from functools import partial
import pickle

def prepare_features_target(data, features, target):
    """Prépare les features et la cible, en gérant les valeurs manquantes"""
    # Sélectionner uniquement les colonnes numériques
    X = data[features].select_dtypes(include=[np.number])
    
    # Gestion des valeurs manquantes
    X = X.fillna(X.mean())
    
    y = data[target]
    # Remplacer les valeurs manquantes dans y par la moyenne
    if y.isnull().any():
        y = y.fillna(y.mean())
    
    return X, y

def select_best_model(X_train, y_train, X_val, y_val):
    """Sélectionne le meilleur modèle parmi plusieurs options"""
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf'),
        'MLPRegressor': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    }
    
    best_mae = float('inf')
    best_model_name = None
    best_model = None
    
    print("\nÉvaluation des différents modèles:")
    print("-" * 50)
    
    for name, model in models.items():
        start_time = time.time()
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédictions sur la validation
        y_pred = model.predict(X_val)
        
        # Calcul des métriques
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val, y_pred)
        
        train_time = time.time() - start_time
        
        print(f"{name}:")
        print(f"  MAE: {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Temps d'entraînement: {train_time:.2f}s")
        
        # Mise à jour du meilleur modèle
        if mae < best_mae:
            best_mae = mae
            best_model_name = name
            best_model = model
    
    print("-" * 50)
    print(f"Meilleur modèle: {best_model_name} (MAE: {best_mae:.4f})")
    
    return best_model, best_model_name

def optimize_model(X_train, y_train, X_val, y_val, model_name):
    """Optimise les hyperparamètres du modèle sélectionné"""
    print("\nOptimisation des hyperparamètres...")
    
    # Définir les grilles de paramètres selon le modèle
    if model_name == 'RandomForest':
        model = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    elif model_name == 'GradientBoosting':
        model = GradientBoostingRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5]
        }
    elif model_name == 'SVR':
        model = SVR()
        param_grid = {
            'kernel': ['linear', 'rbf', 'poly'],
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01]
        }
    elif model_name == 'MLPRegressor':
        model = MLPRegressor(random_state=42, max_iter=1000)
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01],
            'learning_rate': ['constant', 'adaptive']
        }
    else:  # LinearRegression ou défaut
        print("Pas d'optimisation d'hyperparamètres pour ce modèle.")
        return LinearRegression().fit(X_train, y_train)
    
    # GridSearchCV avec validation croisée
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleurs paramètres: {grid_search.best_params_}")
    print(f"Meilleur score MAE: {-grid_search.best_score_:.4f}")
    
    # Évaluer sur l'ensemble de validation
    optimized_model = grid_search.best_estimator_
    y_pred = optimized_model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    print(f"Performance sur validation: MAE={mae:.4f}, RMSE={rmse:.4f}")
    
    return optimized_model

def train_error_corrector(X_val, y_val, y_pred):
    """Entraîne un modèle de correction d'erreur"""
    # Calculer les erreurs
    errors = y_val - y_pred
    
    # Entraîner un modèle pour prédire les erreurs
    error_model = GradientBoostingRegressor(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    error_model.fit(X_val, errors)
    
    # Vérifier l'efficacité du correcteur
    error_predictions = error_model.predict(X_val)
    corrected_predictions = y_pred + error_predictions
    
    original_mae = mean_absolute_error(y_val, y_pred)
    corrected_mae = mean_absolute_error(y_val, corrected_predictions)
    
    print(f"\nCorrecteur d'erreur:")
    print(f"MAE avant correction: {original_mae:.4f}")
    print(f"MAE après correction: {corrected_mae:.4f}")
    print(f"Amélioration: {original_mae - corrected_mae:.4f} ({100 * (original_mae - corrected_mae) / original_mae:.2f}%)")
    
    return error_model

def create_ensemble(X_train, y_train, X_val, y_val):
    """Crée un ensemble de modèles complémentaires"""
    models = [
        GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        RandomForestRegressor(n_estimators=100, max_depth=20, random_state=43),
        MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=44)
    ]
    
    # Entraîner chaque modèle et évaluer
    trained_models = []
    val_predictions = []
    
    for i, model in enumerate(models):
        print(f"\nEntraînement du modèle d'ensemble {i+1}/{len(models)}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        print(f"MAE: {mae:.4f}")
        
        trained_models.append(model)
        val_predictions.append(y_pred)
    
    # Combiner les prédictions (moyenne simple)
    ensemble_predictions = np.mean(val_predictions, axis=0)
    ensemble_mae = mean_absolute_error(y_val, ensemble_predictions)
    
    print(f"\nMAE de l'ensemble: {ensemble_mae:.4f}")
    
    return trained_models

def predict_with_ensemble(models, X):
    """Fait une prédiction en utilisant un ensemble de modèles"""
    predictions = [model.predict(X) for model in models]
    return np.mean(predictions, axis=0)

def test_model(model, X_test, y_test, error_corrector=None, ensemble_models=None):
    """Évalue le modèle sur l'ensemble de test"""
    # Prédictions de base
    if ensemble_models:
        y_pred = predict_with_ensemble(ensemble_models, X_test)
    else:
        y_pred = model.predict(X_test)
    
    # Appliquer le correcteur d'erreur si disponible
    if error_corrector:
        error_pred = error_corrector.predict(X_test)
        y_pred_corrected = y_pred + error_pred
    else:
        y_pred_corrected = y_pred
    
    # Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    mae_corrected = mean_absolute_error(y_test, y_pred_corrected)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rmse_corrected = np.sqrt(mean_squared_error(y_test, y_pred_corrected))
    
    # Calculer l'exactitude à +/- 2°C
    accuracy_2C = np.mean(np.abs(y_test - y_pred_corrected) <= 2)
    
    print("\nPerformance sur l'ensemble de test:")
    if error_corrector:
        print(f"MAE (modèle de base): {mae:.4f}")
        print(f"MAE (avec correction): {mae_corrected:.4f}")
        print(f"RMSE (modèle de base): {rmse:.4f}")
        print(f"RMSE (avec correction): {rmse_corrected:.4f}")
    else:
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {rmse:.4f}")
    
    print(f"Exactitude à +/- 2°C: {accuracy_2C:.2%}")
    
    return mae_corrected, rmse_corrected, accuracy_2C

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
    # Base features + nouvelles caractéristiques créées dans preprocessing.py
    base_features = [
        "minimum_temperature_at_2_metres", "maximum_temperature_at_2_metres",
        "2_metre_relative_humidity", "total_precipitation", "10m_wind_speed",
        "surface_net_solar_radiation", "surface_net_thermal_radiation",
        "surface_solar_radiation_downwards", "surface_latent_heat_flux"
    ]
    
    # Ajouter les caractéristiques temporelles et saisonnières si elles existent
    seasonal_features = [
        "day_of_year_sin", "day_of_year_cos", "month_sin", "month_cos",
        "hour_sin", "hour_cos"
    ]
    
    # Ajouter les caractéristiques d'interaction si elles existent
    interaction_features = [
        "heat_index", "wind_chill", "temp_precip_interaction"
    ]
    
    # Construire la liste complète des features en vérifiant leur présence
    features = base_features.copy()
    
    for feature in seasonal_features + interaction_features:
        if feature in train_data.columns:
            features.append(feature)
    
    print(f"Utilisation de {len(features)} caractéristiques: {features}")
    
    # Liste des cibles à prédire
    targets = [
        "2_metre_temperature",
        "2_metre_dewpoint_temperature",
        "total_precipitation",
        "surface_net_solar_radiation",
        "surface_net_thermal_radiation",
    ]
    
    # Pour chaque cible, entraîner un modèle optimisé
    for target in targets:
        print(f"\n{'='*70}")
        print(f"Entraînement du modèle pour la cible: {target}")
        print(f"{'='*70}")
        
        # Vérifier que la cible existe dans les données
        if target not in train_data.columns:
            print(f"ERREUR: La cible '{target}' n'existe pas dans les données d'entraînement.")
            continue
            
        # Préparer les données
        X_train, y_train = prepare_features_target(train_data, features, target)
        X_val, y_val = prepare_features_target(val_data, features, target)
        X_test, y_test = prepare_features_target(test_data, features, target)
        
        # Standardiser les caractéristiques
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Créer des DataFrames pandas pour les données standardisées
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
        X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
        
        # Sélectionner le meilleur modèle de base
        base_model, best_model_name = select_best_model(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Optimiser les hyperparamètres du modèle sélectionné
        if best_model_name != 'LinearRegression':
            optimized_model = optimize_model(X_train_scaled, y_train, X_val_scaled, y_val, best_model_name)
        else:
            optimized_model = base_model
        
        # Prédictions sur l'ensemble de validation
        val_predictions = optimized_model.predict(X_val_scaled)
        val_mae = mean_absolute_error(y_val, val_predictions)
        
        # Si l'erreur est encore trop grande, utiliser les techniques avancées
        if val_mae > 2.0:
            print(f"\nMAE encore supérieure à 2.0°C ({val_mae:.4f}), application de techniques avancées.")
            
            # Entraîner un correcteur d'erreur
            error_corrector = train_error_corrector(X_val_scaled, y_val, val_predictions)
            
            # Entraîner un ensemble de modèles complémentaires
            ensemble_models = create_ensemble(X_train_scaled, y_train, X_val_scaled, y_val)
        else:
            error_corrector = None
            ensemble_models = None
        
        # Évaluer sur l'ensemble de test
        mae, rmse, accuracy_2C = test_model(
            optimized_model, X_test_scaled, y_test, 
            error_corrector=error_corrector,
            ensemble_models=ensemble_models
        )
        
        # Sauvegarder le modèle et ses composants
        model_filename = os.path.join(models_dir, f"meteo_model_{target.replace(' ', '_')}.pkl")
        
        # Sauvegarder les composants du modèle
        model_components = {
            'target': target,
            'features': features,
            'scaler': scaler,
            'main_model': optimized_model,
            'error_corrector': error_corrector,
            'ensemble_models': ensemble_models,
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'accuracy_2C': accuracy_2C
            }
        }
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model_components, f)
        
        print(f"Modèle sauvegardé dans {model_filename}")