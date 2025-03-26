<?php
require_once 'config.php';
require_once 'API/machine_learning.php';

// Obtenir la connexion à la base de données
$pdo = getDatabaseConnection();

// Variables pour stocker les données
$city = '';
$trend = null;
$prediction = null;
$anomalies = null;

// Récupérer les villes les plus recherchées - DÉPLACÉ ICI, AVANT L'UTILISATION
$topCities = [];
if ($pdo) {
    try {
        $stmt = $pdo->query(
            "SELECT city, COUNT(*) as search_count 
             FROM searches 
             GROUP BY city 
             ORDER BY search_count DESC 
             LIMIT 5"
        );
        $topCities = $stmt->fetchAll(PDO::FETCH_ASSOC);
    } catch (PDOException $e) {
        // Gérer l'erreur silencieusement
    }
}

// Vérifier si une ville a été spécifiée
if (!empty($_GET['city'])) {
    $city = $_GET['city'];
    
    // Obtenir les tendances de température
    $trend = getTemperatureTrend($pdo, $city);
    
    // Obtenir les prédictions météo ajustées
    $prediction = predictWeather($pdo, $city);
    
    // Détecter les anomalies météo
    $anomalies = detectWeatherAnomalies($pdo, $city);
}
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tendances Météorologiques - Y-FLOP</title>
    <style>
        /* styles.css */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Arial', sans-serif;
        }

        body {
            background-color: #f0f4f8;
            color: #333;
            min-height: 100vh;
            padding: 20px;
            background-image: linear-gradient(135deg, #74ebd5, #9face6);
            font-size: 16px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        header {
            text-align: center;
            margin-bottom: 40px;
        }

        h1 {
            font-size: 2.5rem;
            color: #2c3e50;
            margin-bottom: 10px;
            text-align: center;
        }

        h2 {
            font-size: 1.8rem;
            color: #34495e;
            margin-bottom: 20px;
        }

        h3 {
            font-size: 1.4rem;
            color: #34495e;
            margin-bottom: 10px;
        }

        p {
            font-size: 1rem;
            color: #7f8c8d;
            margin-bottom: 15px;
        }

        .info-card {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            padding: 25px;
            margin-bottom: 30px;
        }

        .top-cities {
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
            margin-top: 20px;
        }

        .city-tag {
            background-color: #3498db;
            color: white;
            padding: 8px 15px;
            border-radius: 20px;
            font-size: 0.9rem;
            text-decoration: none;
            transition: background-color 0.3s;
        }

        .city-tag:hover {
            background-color: #2980b9;
        }

        .search-form {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 30px;
        }

        .search-form input[type="text"] {
            width: 100%;
            max-width: 400px;
            padding: 12px;
            margin-bottom: 15px;
            font-size: 1.1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            outline: none;
        }

        .search-form button {
            padding: 12px 20px;
            font-size: 1.1rem;
            color: white;
            background-color: #3498db;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .search-form button:hover {
            background-color: #2980b9;
        }

        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }

        .trend-item {
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 8px;
        }

        .trend-positive {
            color: #27ae60;
            font-weight: bold;
        }

        .trend-negative {
            color: #e74c3c;
            font-weight: bold;
        }

        .trend-neutral {
            color: #f39c12;
            font-weight: bold;
        }

        .anomaly-severe {
            color: #e74c3c;
            font-weight: bold;
        }

        .anomaly-moderate {
            color: #f39c12;
            font-weight: bold;
        }

        .nav-links {
            display: flex;
            justify-content: center;
            margin-bottom: 30px;
        }

        .nav-link {
            padding: 10px 20px;
            text-decoration: none;
            color: #3498db;
            font-weight: bold;
            margin: 0 10px;
            border-radius: 5px;
            transition: background-color 0.3s;
        }

        .nav-link:hover {
            background-color: rgba(52, 152, 219, 0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Y-FLOP Tendances Météorologiques</h1>
            <p>Analyse des données météorologiques et apprentissage automatique</p>
        </header>

        <div class="nav-links">
            <a href="index.php" class="nav-link">Météo Actuelle</a>
            <a href="trends.php" class="nav-link">Tendances</a>
        </div>

        <div class="info-card">
            <h2>Villes les plus recherchées</h2>
            <div class="top-cities">
                <?php foreach ($topCities as $topCity): ?>
                    <a href="?city=<?= urlencode($topCity['city']) ?>" class="city-tag">
                        <?= htmlspecialchars($topCity['city']) ?> (<?= $topCity['search_count'] ?>)
                    </a>
                <?php endforeach; ?>
            </div>
        </div>

        <div class="search-form">
            <h2>Analyser les tendances pour une ville</h2>
            <form method="GET">
                <input type="text" name="city" placeholder="Entrez une ville" required value="<?= htmlspecialchars($city) ?>">
                <button type="submit">Analyser</button>
            </form>
        </div>

        <?php if ($city): ?>
            <div class="info-card">
                <h2>Analyse pour <?= htmlspecialchars($city) ?></h2>
                
                <?php if (isset($trend['error'])): ?>
                    <p><?= $trend['error'] ?></p>
                <?php elseif (isset($trend['trend'])): ?>
                    <h3>Tendance de température</h3>
                    <p>Température actuelle: <strong><?= $trend['current'] ?> °C</strong></p>
                    <p>
                        Tendance: 
                        <span class="<?= $trend['trend'] > 0 ? 'trend-positive' : ($trend['trend'] < 0 ? 'trend-negative' : 'trend-neutral') ?>">
                            <?= $trend['trend'] > 0 ? '+' : '' ?><?= round($trend['trend'], 1) ?> °C (<?= $trend['trend_direction'] ?>)
                        </span>
                    </p>
                    <p>Basé sur <?= $trend['data_points'] ?> points de données</p>
                <?php endif; ?>
                
                <?php if (isset($prediction['error'])): ?>
                    <p><?= $prediction['error'] ?></p>
                <?php elseif (isset($prediction['adjusted_forecasts'])): ?>
                    <h3>Prévisions ajustées</h3>
                    <p>Qualité des données: <strong><?= $prediction['data_quality'] ?></strong></p>
                    
                    <div class="grid">
                        <?php foreach ($prediction['adjusted_forecasts'] as $forecast): ?>
                            <div class="trend-item">
                                <h4><?= date('d/m/Y', strtotime($forecast['forecast_date'])) ?></h4>
                                <p>Prévision standard: <strong><?= $forecast['temperature'] ?> °C</strong></p>
                                <p>Prévision ajustée: <strong><?= $forecast['adjusted_temperature'] ?> °C</strong></p>
                                <p>Confiance: <strong><?= $forecast['confidence'] ?></strong></p>
                            </div>
                        <?php endforeach; ?>
                    </div>
                <?php endif; ?>
                
                <?php if (isset($anomalies['error'])): ?>
                    <p><?= $anomalies['error'] ?></p>
                <?php elseif (isset($anomalies['anomalies'])): ?>
                    <h3>Anomalies détectées</h3>
                    <?php if (count($anomalies['anomalies']) > 0): ?>
                        <div class="grid">
                            <?php foreach ($anomalies['anomalies'] as $anomaly): ?>
                                <div class="trend-item">
                                    <h4><?= date('d/m/Y', strtotime($anomaly['date'])) ?></h4>
                                    <p>Température: <strong><?= $anomaly['temperature'] ?> °C</strong></p>
                                    <p>
                                        Sévérité: 
                                        <span class="anomaly-<?= $anomaly['severity'] === 'élevée' ? 'severe' : 'moderate' ?>">
                                            <?= $anomaly['severity'] ?>
                                        </span>
                                    </p>
                                </div>
                            <?php endforeach; ?>
                        </div>
                    <?php else: ?>
                        <p>Aucune anomalie détectée.</p>
                    <?php endif; ?>
                <?php endif; ?>
            </div>
        <?php endif; ?>
        
        <div class="info-card">
            <h2>À propos de l'apprentissage automatique</h2>
            <p>Notre système utilise les données collectées lors des recherches des utilisateurs pour améliorer les prévisions météorologiques. Plus vous utilisez notre application, plus nos prédictions deviennent précises.</p>
            <p>Le système analyse les tendances historiques, détecte les anomalies, et ajuste les prévisions en fonction des données réelles observées dans votre région.</p>
        </div>
    </div>
</body>
</html>