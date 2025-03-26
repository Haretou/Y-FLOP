<?php
// Activer l'affichage des erreurs
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

require_once 'API/api.php';
require_once 'CRUD/read.php';
require_once 'CRUD/create.php';

$weather = null;
$forecast = null;

if (!empty($_POST['city'])) {
    $city = $_POST['city'];
    
    // Récupérer les données météo actuelles
    $weather = fetchWeatherData('current.json', ['q' => $city]);
    
    // Débogage - vérifier les données météo actuelles
    error_log("Données météo actuelles pour $city: " . json_encode($weather));
    
    // Récupérer les prévisions sur 7 jours
    $forecastData = fetchWeeklyForecast($city);
    
    // Débogage - vérifier les données de prévision
    error_log("Prévisions pour $city: " . json_encode($forecastData));
    
    if (isset($weather['current'])) {
        // Traduire les conditions météorologiques
        $conditions = [
            'Sunny' => 'Ensoleillé',
            'Partly cloudy' => 'Partiellement nuageux',
            'Cloudy' => 'Nuageux',
            'Overcast' => 'Couvert',
            'Mist' => 'Brume',
            'Patchy rain possible' => 'Pluie éparse possible',
            'Patchy snow possible' => 'Neige éparse possible',
            'Patchy sleet possible' => 'Neige fondue éparse possible',
            'Patchy freezing drizzle possible' => 'Bruine verglaçante éparse possible',
            'Thundery outbreaks possible' => 'Orages possibles',
            'Light rain' => 'Pluie légère',
            'Moderate rain' => 'Pluie modérée',
            'Heavy rain' => 'Forte pluie',
            'Light snow' => 'Neige légère',
            'Moderate snow' => 'Neige modérée',
            'Heavy snow' => 'Forte neige',
            'Clear' => 'Clair'
        ];
        
        if (isset($weather['current']['condition']['text'])) {
            $conditionText = $weather['current']['condition']['text'];
            if (array_key_exists($conditionText, $conditions)) {
                $weather['current']['condition']['text'] = $conditions[$conditionText];
            }
        }
        
        // Enregistrer la recherche dans la base de données
        $result = saveSearch($city, $weather['current']['temp_c'], $weather['current']['condition']['text']);
        // Débogage - vérifier si l'enregistrement a fonctionné
        error_log("Enregistrement de la recherche pour $city: " . ($result ? "succès" : "échec"));
        
        // Traiter et enregistrer les prévisions
        if (isset($forecastData['forecast']['forecastday'])) {
            $forecast = $forecastData['forecast']['forecastday'];
            
            // Enregistrer chaque jour de prévision dans la base de données
            foreach ($forecast as $day) {
                $date = $day['date'];
                $temp = $day['day']['avgtemp_c'];
                $conditionText = $day['day']['condition']['text'];
                
                // Traduire la condition si elle existe dans notre tableau
                if (array_key_exists($conditionText, $conditions)) {
                    $conditionText = $conditions[$conditionText];
                }
                
                $humidity = $day['day']['avghumidity'];
                $precipitation = $day['day']['totalprecip_mm'];
                $wind = $day['day']['maxwind_kph'];
                
                $forecastResult = saveForecast($city, $date, $temp, $conditionText, $humidity, $precipitation, $wind);
                // Débogage - vérifier si l'enregistrement des prévisions a fonctionné
                error_log("Enregistrement prévision pour $city ($date): " . ($forecastResult ? "succès" : "échec"));
            }
        } else {
            error_log("Pas de données de prévision pour $city");
        }
    } else {
        error_log("Pas de données météo actuelles pour $city");
    }
}
?>
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Y-FLOP Météo</title>
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
            background-image: linear-gradient(135deg, #74ebd5, #9face6); /* Gradient background */
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
            margin-bottom: 10px;
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

        form {
            display: flex;
            flex-direction: column;
            align-items: center;
            background-color: #ffffff;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            margin-bottom: 30px;
            width: 100%;
            max-width: 400px;
            margin: 0 auto;
        }

        input[type="text"] {
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            font-size: 1.1rem;
            border: 1px solid #ddd;
            border-radius: 8px;
            outline: none;
            transition: border-color 0.3s ease;
        }

        input[type="text"]:focus {
            border-color: #3498db;
        }

        button[type="submit"] {
            padding: 12px 20px;
            font-size: 1.1rem;
            color: white;
            background-color: #3498db;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            width: 100%;
        }

        button[type="submit"]:hover {
            background-color: #2980b9;
        }

        .weather-info {
            background-color: #ecf0f1;
            padding: 25px;
            border-radius: 10px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
            width: 100%;
            max-width: 800px;
            text-align: center;
            font-size: 1.2rem;
            margin: 20px auto;
        }

        .weather-info p {
            margin-bottom: 15px;
            font-weight: bold;
            color: #34495e;
        }

        .weather-info h2 {
            color: #2980b9;
            margin-bottom: 20px;
        }

        .forecast-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            margin-top: 40px;
        }

        .forecast-day {
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
            padding: 15px;
            margin-bottom: 20px;
            width: calc(25% - 15px);
            text-align: center;
        }

        .forecast-day h3 {
            color: #2980b9;
            margin-bottom: 10px;
        }

        .forecast-day p {
            margin-bottom: 8px;
            color: #34495e;
        }

        @media (max-width: 900px) {
            .forecast-day {
                width: calc(33.33% - 15px);
            }
        }

        @media (max-width: 700px) {
            .forecast-day {
                width: calc(50% - 10px);
            }
        }

        @media (max-width: 600px) {
            h1 {
                font-size: 2rem;
            }

            form {
                width: 100%;
                padding: 20px;
            }

            input[type="text"] {
                width: 100%;
            }

            button[type="submit"] {
                width: 100%;
            }

            .weather-info {
                max-width: 100%;
                padding: 20px;
            }
            
            .forecast-day {
                width: 100%;
            }
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
            <h1>Y-FLOP Météo</h1>
            <p>Prévisions météorologiques détaillées et historique des données</p>
        </header>

        <div class="nav-links">
            <a href="index.php" class="nav-link">Météo Actuelle</a>
            <a href="trends.php" class="nav-link">Tendances</a>
        </div>

        <form method="POST">
        <div style="background: #fff; padding: 10px; margin: 20px; border: 1px solid #000;">
    <h3>DEBUG INFO:</h3>
    <?php if (!empty($_POST['city'])): ?>
        <p>City: <?= htmlspecialchars($_POST['city']) ?></p>
        <p>API Response (Current): <?= isset($weather) ? 'OK' : 'Failed' ?></p>
        <p>API Response (Forecast): <?= isset($forecastData) ? 'OK' : 'Failed' ?></p>
        <?php if (isset($weather['current'])): ?>
            <p>Temp: <?= $weather['current']['temp_c'] ?> °C</p>
            <p>Save Result: <?= var_export($result, true) ?></p>
        <?php endif; ?>
        <?php if (isset($forecastData['forecast']['forecastday'])): ?>
            <p>Number of forecast days: <?= count($forecastData['forecast']['forecastday']) ?></p>
        <?php else: ?>
            <p>No forecast data received!</p>
        <?php endif; ?>
    <?php else: ?>
        <p>No city submitted yet</p>
    <?php endif; ?>
</div>
            <input type="text" name="city" placeholder="Entrez une ville" required>
            <button type="submit">Rechercher</button>
        </form>

        <?php if ($weather): ?>
            <div class="weather-info">
                <h2>Météo actuelle pour <?= htmlspecialchars($city) ?></h2>
                <p>Température : <?= $weather['current']['temp_c'] ?> °C</p>
                <p>Condition : <?= $weather['current']['condition']['text'] ?></p>
                <p>Humidité : <?= $weather['current']['humidity'] ?>%</p>
                <p>Précipitations : <?= isset($weather['current']['precip_mm']) ? $weather['current']['precip_mm'] . " mm" : "Non disponible" ?></p>
                <p>Vent : <?= isset($weather['current']['wind_kph']) ? $weather['current']['wind_kph'] . " km/h" : "Non disponible" ?></p>
                <p>Pression : <?= isset($weather['current']['pressure_mb']) ? $weather['current']['pressure_mb'] . " mb" : "Non disponible" ?></p>
            </div>

            <?php if ($forecast): ?>
                <div class="weather-info">
                    <h2>Prévisions sur 7 jours pour <?= htmlspecialchars($city) ?></h2>
                    <div class="forecast-container">
                        <?php foreach ($forecast as $index => $day): ?>
                            <div class="forecast-day">
                                <h3><?= date('d/m', strtotime($day['date'])) ?></h3>
                                <p><?= date('l', strtotime($day['date'])) ?></p>
                                <p>Temp. Min: <?= $day['day']['mintemp_c'] ?> °C</p>
                                <p>Temp. Max: <?= $day['day']['maxtemp_c'] ?> °C</p>
                                <p>Condition: <?= isset($conditions[$day['day']['condition']['text']]) ? $conditions[$day['day']['condition']['text']] : $day['day']['condition']['text'] ?></p>
                                <p>Humidité: <?= $day['day']['avghumidity'] ?>%</p>
                                <p>Précip.: <?= $day['day']['totalprecip_mm'] ?> mm</p>
                                <p>Vent: <?= $day['day']['maxwind_kph'] ?> km/h</p>
                            </div>
                        <?php endforeach; ?>
                    </div>
                </div>
            <?php endif; ?>
        <?php endif; ?>
        
        <!-- Section des recherches récentes pour l'apprentissage automatique -->
        <div class="weather-info">
            <h2>Tendances et apprentissage automatique</h2>
            <p>Notre application utilise vos recherches pour améliorer nos prédictions météorologiques grâce à l'apprentissage automatique.</p>
            
            <?php
            // Obtenir les recherches récentes
            $recentSearches = getRecentSearches($pdo, 5);
            if (!empty($recentSearches)): ?>
                <h3>Recherches récentes</h3>
                <div class="forecast-container">
                    <?php foreach ($recentSearches as $search): ?>
                        <div class="forecast-day">
                            <h3><?= htmlspecialchars($search['city']) ?></h3>
                            <p>Température: <?= $search['temperature'] ?> °C</p>
                            <p>Condition: <?= $search['condition'] ?></p>
                        </div>
                    <?php endforeach; ?>
                </div>
            <?php endif; ?>
        </div>
    </div>
</body>
</html>