# Y-FLOP Predict - Application de Prévisions Météorologiques

Y-FLOP Predict est une application web de prévisions météorologiques qui permet de consulter la météo actuelle et les prévisions pour différentes communes françaises, ainsi que de faire des prédictions personnalisées basées sur un modèle d'apprentissage automatique.

## 🌦️ Caractéristiques

- **Consultation de la météo actuelle** pour n'importe quelle commune en France
- **Prévisions sur 5 jours** avec graphique d'évolution des températures
- **Recherche de communes** avec auto-complétion
- **Prédictions personnalisées** basées sur un modèle d'apprentissage automatique
- **Interface responsive** adaptée à tous les appareils
- **Mode sombre** pour une utilisation confortable de nuit
- **Animations météorologiques** pour une expérience utilisateur immersive

## 📋 Prérequis

- Python 3.9 ou supérieur
- Flask et ses dépendances
- Pandas, NumPy, Scikit-learn et Joblib pour le traitement des données et le machine learning

## 🔧 Installation

1. Cloner le dépôt :
   ```bash
   git clone https://github.com/yourusername/y-flop-predict.git
   cd y-flop-predict
   ```

2. Créer et activer un environnement virtuel (recommandé) :
   ```bash
   python -m venv venv
   source venv/bin/activate  # Sur Windows : venv\Scripts\activate
   ```

3. Installer les dépendances :
   ```bash
   pip install -r requirement.txt
   ```

4. Préparer les données et entraîner le modèle (si les données et modèles ne sont pas inclus) :
   ```bash
   python preprocessing.py
   python train_model.py
   ```

## 🚀 Démarrage

1. Démarrer l'application Flask :
   ```bash
   python app.py
   ```

2. Ouvrir votre navigateur et accéder à l'URL :
   ```
   http://127.0.0.1:5000/
   ```

## 📁 Structure du Projet

```
y-flop-predict/
├── app.py                 # Application principale Flask
├── preprocessing.py       # Script de prétraitement des données
├── train_model.py         # Script d'entraînement du modèle
├── requirement.txt        # Dépendances du projet
├── data/                  # Dossier contenant les données
│   ├── meteo-0025.csv     # Données météo brutes
│   ├── train_data.csv     # Données d'entraînement
│   ├── val_data.csv       # Données de validation
│   └── test_data.csv      # Données de test
├── models/                # Dossier contenant les modèles entraînés
│   └── meteo_model.pkl    # Modèle de prédiction météo
├── static/                # Fichiers statiques
│   ├── css/
│   │   └── style.css      # Feuille de style CSS
│   └── js/
│       └── script.js      # JavaScript pour l'interaction
└── templates/             # Templates HTML
    └── index.html         # Page principale
```

## 🛠️ Technologies Utilisées

- **Backend** : Flask (Python)
- **Frontend** : HTML, CSS, JavaScript
- **Data Processing** : Pandas, NumPy
- **Machine Learning** : Scikit-learn
- **Visualisation** : Chart.js

## 📊 Modèle de Prédiction

L'application utilise un modèle de régression linéaire entraîné sur des données météorologiques historiques pour prédire plusieurs paramètres météorologiques :

- Température à 2 mètres du sol
- Point de rosée
- Précipitations totales
- Rayonnement solaire net
- Rayonnement thermique net

Les caractéristiques utilisées pour l'entraînement comprennent :
- Température minimale et maximale
- Humidité relative
- Précipitations
- Vitesse du vent
- Rayonnement solaire et thermique
- Flux de chaleur latente

## 🔄 Flux de Données

1. Les données brutes sont prétraitées avec `preprocessing.py`
2. Le modèle est entraîné avec `train_model.py`
3. L'application utilise le modèle via `app.py` pour fournir des prédictions

## 🌐 API

L'application expose plusieurs points d'API REST :

- `GET /` : Page principale
- `GET /search_city?q=<query>` : Recherche de communes
- `GET /city_weather/<commune_name>` : Météo actuelle d'une commune
- `GET /forecast/<commune_name>` : Prévisions pour une commune
- `POST /predict` : Prédiction personnalisée basée sur les paramètres fournis

## ✨ Fonctionnalités à Venir

- Intégration de sources de données météo en temps réel
- Prévisions horaires détaillées
- Alertes météorologiques personnalisées
- Cartographie interactive des données météo
- Support multilingue
