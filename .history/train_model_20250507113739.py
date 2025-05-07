# Importations
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import yaml
import argparse

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path='config.yaml'):
    """Charge la configuration à partir d'un fichier YAML."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.warning(f"Fichier de configuration {config_path} non trouvé. Utilisation des valeurs par défaut.")
        return {
            'data_dir': 'data',
            'models_dir': 'models',
            'results_dir': 'results',
            'features': [
                "minimum_temperature_at_2_metres", "maximum_temperature_at_2_metres",
                "2_metre_relative_humidity", "total_precipitation", "10m_wind_speed",
                "surface_net_solar_radiation", "surface_net_thermal_radiation",
                "surface_solar_radiation_downwards", "surface_latent_heat_flux"
            ],
            'targets': [
                "2_metre_temperature",
                "2_metre_dewpoint_temperature",
                "total_precipitation",
                "surface_net_solar_radiation",
                "surface_net_thermal_radiation",
            ],
            'model_type': 'linear'  # options: 'linear', 'rf', 'gb'
        }

def prepare_features_target(data, features, target, scale=True):
    """
    Prépare les caractéristiques et la cible pour l'entraînement.
    
    Args:
        data: DataFrame contenant les données
        features: Liste des caractéristiques à utiliser
        target: Variable cible à prédire
        scale: Booléen indiquant si les caractéristiques doivent être normalisées
        
    Returns:
        X: Caractéristiques préparées
        y: Variable cible
        scaler: Scaler utilisé (ou None si scale=False)
    """
    # Vérifier les valeurs manquantes
    missing_values = data[features].isna().sum()
    if missing_values.sum() > 0:
        logger.warning(f"Valeurs manquantes détectées dans les caractéristiques:\n{missing_values[missing_values > 0]}")
        # Imputation simple des valeurs manquantes avec la médiane
        data[features] = data[features].fillna(data[features].median())
    
    X = data[features].select_dtypes(include=[np.number])
    y = data[target]
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    
    return X, y, scaler

def get_model(model_type='linear'):
    """
    Retourne un modèle en fonction du type spécifié.
    
    Args:
        model_type: Type de modèle ('linear', 'rf', 'gb')
        
    Returns:
        Un modèle sklearn initialisé
    """
    if model_type.lower() == 'rf':
        return RandomForestRegressor(n_estimators=100, random_state=42)
    elif model_type.lower() == 'gb':
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:  # Par défaut: modèle linéaire
        return LinearRegression()

def train_and_evaluate_model(X_train, y_train, X_val, y_val, model_type='linear'):
    """
    Entraîne et évalue un modèle.
    
    Args:
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
        model_type: Type de modèle à utiliser
        
    Returns:
        model: Modèle entraîné
        metrics: Dictionnaire contenant les métriques d'évaluation
    """
    # Initialisation et entraînement du modèle
    model = get_model(model_type)
    logger.info(f"Entraînement du modèle {model.__class__.__name__}")
    
    # Enregistrer le temps d'entraînement
    start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = (datetime.now() - start_time).total_seconds()
    
    # Prédictions sur l'ensemble d'entraînement et de validation
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Calcul des métriques d'évaluation
    metrics = {
        'train': {
            'mae': mean_absolute_error(y_train, y_train_pred),
            'mse': mean_squared_error(y_train, y_train_pred),
            'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'r2': r2_score(y_train, y_train_pred)
        },
        'val': {
            'mae': mean_absolute_error(y_val, y_val_pred),
            'mse': mean_squared_error(y_val, y_val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'r2': r2_score(y_val, y_val_pred)
        },
        'training_time': training_time
    }
    
    # Affichage des métriques
    logger.info(f"Performance sur l'ensemble d'entraînement:")
    logger.info(f"  MAE: {metrics['train']['mae']:.4f}")
    logger.info(f"  MSE: {metrics['train']['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['train']['rmse']:.4f}")
    logger.info(f"  R²: {metrics['train']['r2']:.4f}")
    
    logger.info(f"Performance sur l'ensemble de validation:")
    logger.info(f"  MAE: {metrics['val']['mae']:.4f}")
    logger.info(f"  MSE: {metrics['val']['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['val']['rmse']:.4f}")
    logger.info(f"  R²: {metrics['val']['r2']:.4f}")
    logger.info(f"Temps d'entraînement: {training_time:.2f} secondes")
    
    return model, metrics

def test_model(model, X_test, y_test):
    """
    Teste un modèle sur des données de test.
    
    Args:
        model: Modèle entraîné
        X_test, y_test: Données de test
        
    Returns:
        metrics: Dictionnaire contenant les métriques d'évaluation
    """
    # Prédictions sur l'ensemble de test
    y_test_pred = model.predict(X_test)
    
    # Calcul des métriques
    metrics = {
        'mae': mean_absolute_error(y_test, y_test_pred),
        'mse': mean_squared_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2': r2_score(y_test, y_test_pred),
        'predictions': y_test_pred,
        'actual': y_test
    }
    
    logger.info(f"Performance sur l'ensemble de test:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  MSE: {metrics['mse']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    
    return metrics

def plot_results(y_true, y_pred, target_name, output_path):
    """
    Crée et sauvegarde des visualisations des résultats.
    
    Args:
        y_true: Valeurs réelles
        y_pred: Prédictions
        target_name: Nom de la variable cible
        output_path: Chemin pour sauvegarder les visualisations
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Créer un dataframe pour faciliter la visualisation
    results_df = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    # Figure avec plusieurs sous-graphiques
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Graphique de dispersion
    axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
    axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    axes[0, 0].set_xlabel('Valeurs réelles')
    axes[0, 0].set_ylabel('Prédictions')
    axes[0, 0].set_title(f'Prédictions vs Valeurs réelles pour {target_name}')
    
    # 2. Histogramme des erreurs
    errors = y_pred - y_true
    axes[0, 1].hist(errors, bins=30, alpha=0.7)
    axes[0, 1].axvline(x=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Erreur de prédiction')
    axes[0, 1].set_ylabel('Fréquence')
    axes[0, 1].set_title('Distribution des erreurs')
    
    # 3. Box plot des valeurs réelles et prédites
    axes[1, 0] = sns.boxplot(data=results_df, ax=axes[1, 0])
    axes[1, 0].set_title('Comparaison des distributions')
    
    # 4. Residual plot
    axes[1, 1].scatter(y_pred, errors, alpha=0.5)
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Prédictions')
    axes[1, 1].set_ylabel('Erreurs')
    axes[1, 1].set_title('Graphique des résidus')
    
    # Ajuster la mise en page
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.info(f"Visualisations sauvegardées dans {output_path}")

def save_model_with_metadata(model, scaler, features, metrics, target, output_path):
    """
    Sauvegarde le modèle avec ses métadonnées.
    
    Args:
        model: Modèle entraîné
        scaler: Scaler utilisé pour normaliser les données
        features: Liste des caractéristiques utilisées
        metrics: Métriques d'évaluation
        target: Nom de la variable cible
        output_path: Chemin pour sauvegarder le modèle
    """
    # Créer le répertoire de sortie s'il n'existe pas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarder le modèle et ses métadonnées
    model_data = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'metrics': metrics,
        'target': target,
        'timestamp': datetime.now().isoformat(),
        'model_type': model.__class__.__name__
    }
    
    joblib.dump(model_data, output_path)
    logger.info(f"Modèle sauvegardé dans {output_path}")

def main(config_path=None):
    """Fonction principale d'entraînement des modèles."""
    # Charger la configuration
    config = load_config(config_path) if config_path else load_config()
    
    # Extraire les paramètres de configuration
    data_dir = config['data_dir']
    models_dir = config['models_dir']
    results_dir = config['results_dir']
    features = config['features']
    targets = config['targets']
    model_type = config.get('model_type', 'linear')
    
    # Créer les répertoires s'ils n'existent pas
    for directory in [models_dir, results_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Répertoire créé: {directory}")
    
    # Charger les données prétraitées
    try:
        train_data = pd.read_csv(os.path.join(data_dir, "train_data.csv"))
        val_data = pd.read_csv(os.path.join(data_dir, "val_data.csv"))
        test_data = pd.read_csv(os.path.join(data_dir, "test_data.csv"))
        logger.info(f"Données chargées avec succès: {train_data.shape[0]} échantillons d'entraînement, "
                   f"{val_data.shape[0]} échantillons de validation, {test_data.shape[0]} échantillons de test")
    except FileNotFoundError as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        return
    
    # Entraîner un modèle pour chaque cible
    for target in targets:
        logger.info(f"\n{'='*50}")
        logger.info(f"Entraînement du modèle pour la cible: {target}")
        logger.info(f"{'='*50}")
        
        # Vérifier que la cible existe dans les données
        if target not in train_data.columns:
            logger.error(f"La cible '{target}' n'existe pas dans les données d'entraînement.")
            continue
            
        # Préparer les données
        X_train, y_train, scaler = prepare_features_target(train_data, features, target, scale=True)
        X_val, y_val, _ = prepare_features_target(val_data, features, target, scale=False)
        X_test, y_test, _ = prepare_features_target(test_data, features, target, scale=False)
        
        # Appliquer le même scaling aux données de validation et de test
        if scaler:
            X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
            X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        
        # Entraîner et évaluer le modèle
        model, train_val_metrics = train_and_evaluate_model(X_train, y_train, X_val, y_val, model_type)
        
        # Tester le modèle
        test_metrics = test_model(model, X_test, y_test)
        
        # Générer des visualisations
        plot_path = os.path.join(results_dir, f"{target.replace(' ', '_')}_results.png")
        plot_results(test_metrics['actual'], test_metrics['predictions'], target, plot_path)
        
        # Fusionner les métriques
        all_metrics = {
            'training': train_val_metrics,
            'test': test_metrics
        }
        
        # Sauvegarder le modèle avec ses métadonnées
        model_filename = f"meteo_model_{target.replace(' ', '_')}.pkl"
        model_path = os.path.join(models_dir, model_filename)
        save_model_with_metadata(model, scaler, features, all_metrics, target, model_path)

if __name__ == "__main__":
    # Configurer l'analyse des arguments en ligne de commande
    parser = argparse.ArgumentParser(description='Entraînement de modèles météorologiques')
    parser.add_argument('--config', type=str, help='Chemin vers le fichier de configuration YAML')
    args = parser.parse_args()
    
    main(args.config)