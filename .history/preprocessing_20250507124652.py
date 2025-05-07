# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
import re

def load_and_clean_data(file_path):
    # Charger le fichier CSV
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    
    # Nettoyer les données
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Convertir les dates si nécessaire
    if 'forecast_base' in df.columns:
        df['forecast_base'] = pd.to_datetime(df['forecast_base'], errors='coerce')
    if 'forecast_timestamp' in df.columns:
        df['forecast_timestamp'] = pd.to_datetime(df['forecast_timestamp'], errors='coerce')
    
    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
    return df

def add_seasonal_features(df):
    """Ajoute des caractéristiques temporelles et saisonnières"""
    # Utiliser la colonne timestamp appropriée
    date_col = None
    if 'forecast_timestamp' in df.columns:
        date_col = 'forecast_timestamp'
    elif 'forecast_base' in df.columns:
        date_col = 'forecast_base'
    
    if date_col:
        # S'assurer que c'est bien un datetime
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extraire des caractéristiques temporelles
        df['day_of_year'] = df[date_col].dt.dayofyear
        df['month'] = df[date_col].dt.month
        df['day_of_week'] = df[date_col].dt.dayofweek
        df['hour'] = df[date_col].dt.hour
        
        # Transformations cycliques pour capturer la saisonnalité
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year']/365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year']/365.25)
        df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
        df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour']/24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour']/24)
    
    return df

def add_geographic_features(df):
    """Ajoute des caractéristiques géographiques"""
    # Si position est disponible, extraire lat/lon
    if 'position' in df.columns:
        # Créer des colonnes pour lat/lon si elles n'existent pas déjà
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            # Utiliser regex pour extraire lat/lon de la colonne position
            df['latitude'] = df['position'].apply(
                lambda x: float(re.search(r'([\d\.-]+),\s*([\d\.-]+)', str(x)).group(1)) 
                if isinstance(x, str) and re.search(r'([\d\.-]+),\s*([\d\.-]+)', str(x)) 
                else np.nan
            )
            df['longitude'] = df['position'].apply(
                lambda x: float(re.search(r'([\d\.-]+),\s*([\d\.-]+)', str(x)).group(2)) 
                if isinstance(x, str) and re.search(r'([\d\.-]+),\s*([\d\.-]+)', str(x)) 
                else np.nan
            )
        
        # Ajouter une caractéristique de distance à la mer si possible
        # Ceci est un exemple simplifié - vous devriez adapter en fonction de vos données
        if 'code_commune' in df.columns:
            # Vérifier si le code commence par certains départements côtiers
            coastal_depts = ['06', '11', '13', '14', '17', '22', '29', '30', '33', 
                            '34', '35', '40', '44', '50', '56', '59', '62', '64', 
                            '66', '76', '80', '83', '85']
            
            df['is_coastal'] = df['code_commune'].apply(
                lambda x: 1 if str(x)[:2] in coastal_depts else 0
            )
    
    return df

def add_interaction_features(df):
    """Ajoute des caractéristiques d'interaction entre variables"""
    # Interactions météorologiques connues
    if all(col in df.columns for col in ['2_metre_relative_humidity', '2_metre_temperature']):
        # Indice de chaleur (Heat Index) - approximation simplifiée
        df['heat_index'] = df.apply(
            lambda row: row['2_metre_temperature'] + 0.05 * row['2_metre_relative_humidity'] 
            if row['2_metre_temperature'] > 20 else row['2_metre_temperature'],
            axis=1
        )
    
    if all(col in df.columns for col in ['2_metre_temperature', '10m_wind_speed']):
        # Indice de refroidissement éolien (Wind Chill)
        df['wind_chill'] = df.apply(
            lambda row: row['2_metre_temperature'] - (row['10m_wind_speed'] / 5) 
            if row['2_metre_temperature'] < 10 and row['10m_wind_speed'] > 3 
            else row['2_metre_temperature'],
            axis=1
        )
    
    # Interaction température-précipitation
    if all(col in df.columns for col in ['2_metre_temperature', 'total_precipitation']):
        df['temp_precip_interaction'] = df['2_metre_temperature'] * df['total_precipitation']
    
    return df

def enrich_data(df):
    """Fonction principale pour enrichir les données"""
    df = add_seasonal_features(df)
    df = add_geographic_features(df)
    df = add_interaction_features(df)
    return df

def split_data(df):
    # Diviser en train (70%) et test (30%)
    train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
    
    # Diviser le test en validation (15%) et test final (15%)
    val_data, test_data = train_test_split(test_data, test_size=0.5, random_state=42)
    
    print(f"Taille des ensembles :\n- Train: {len(train_data)}\n- Validation: {len(val_data)}\n- Test: {len(test_data)}")
    
    return train_data, val_data, test_data

if __name__ == "__main__":
    data_dir = "data"
    file_path = os.path.join(data_dir, "meteo-0025.csv")
    
    print("Chargement et nettoyage des données...")
    df = load_and_clean_data(file_path)
    
    print("Enrichissement des données avec des caractéristiques supplémentaires...")
    df = enrich_data(df)
    
    print("Division des données en ensembles d'entraînement, validation et test...")
    train_data, val_data, test_data = split_data(df)
    
    # Sauvegarder les ensembles de données préparés
    train_data.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(data_dir, "val_data.csv"), index=False)
    test_data.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)
    
    print("Prétraitement terminé. Données enrichies sauvegardées dans le répertoire", data_dir)