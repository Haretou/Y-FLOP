# preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean_data(file_path):
    # Charger le fichier CSV
    df = pd.read_csv(file_path, delimiter=';', encoding='utf-8')
    
    # Nettoyer les données
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    
    # Convertir les dates si nécessaire
    if 'forecast_base' in df.columns:
        df['forecast_base'] = pd.to_datetime(df['forecast_base'], errors='coerce')
    
    # Normaliser les noms de colonnes
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    
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
    
    df = load_and_clean_data(file_path)
    train_data, val_data, test_data = split_data(df)
    
    # Sauvegarder les ensembles de données préparés
    train_data.to_csv(os.path.join(data_dir, "train_data.csv"), index=False)
    val_data.to_csv(os.path.join(data_dir, "val_data.csv"), index=False)
    test_data.to_csv(os.path.join(data_dir, "test_data.csv"), index=False)